from typing import List, Optional, Tuple, TypedDict
from loss import MultiCEFocalLoss
import torch
import logging
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from opt_einsum import contract
from long_seq import process_long_input


logger = logging.getLogger(__name__)


class TInfoDict(TypedDict):
    event_idx: int
    event_ids: List[int]
    role_idxs: List[int]
    words_num: int


class MyBertmodel(BertPreTrainedModel):
    def __init__(
        self,
        config,
        lambda_boundary=0,
        event_embedding_size=200,
        emb_size=768,
        group_size=64,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.emb_size = emb_size
        self.group_size = group_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_func = nn.ReLU()
        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.len_embedding = nn.Embedding(config.len_size, config.len_dim)
        if event_embedding_size > 0:
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 5 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels),
            )
        else:
            self.event_embedding = None
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels),
            )

        # boundary
        self.lambda_boundary = lambda_boundary
        if self.lambda_boundary > 0:
            self.start_classifier = nn.Linear(config.hidden_size, 2)
            self.end_classifier = nn.Linear(config.hidden_size, 2)

        # positive weight
        pos_loss_weight = getattr(config, "pos_loss_weight", None)
        self.focal_loss = MultiCEFocalLoss(self.num_labels)
        self.pos_loss_weight = torch.tensor(
            [pos_loss_weight for _ in range(self.num_labels)]
        )
        self.pos_loss_weight[0] = 1
        self.begin_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.end_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.context_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.init_weights()

    def encode(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        head_mask,
        inputs_embeds,
        position_ids,
        output_hidden_states,
        return_dict,
    ):
        start_tokens = [101]
        end_tokens = [102]
        sequence_output, attention = process_long_input(
            self.bert,
            input_ids,
            token_type_ids,
            attention_mask,
            start_tokens,
            end_tokens,
            head_mask,
            inputs_embeds,
            position_ids,
            output_hidden_states,
            return_dict,
        )
        return sequence_output, attention

    def select_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B x num
        Returns:
            B x num x dim
        """
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()

        # print("B:", B)
        # print("L:", L)
        # print("dim:", dim)
        # print("num:", num)
        shift = (
            (torch.arange(B).unsqueeze(-1).expand(-1, num) * L)
            .contiguous()
            .view(-1)
            .to(batch_rep.device)
        )
        # print(shift)
        token_pos = token_pos.contiguous().view(-1)
        # print(token_pos.shape, token_pos.max())
        token_pos = token_pos + shift
        # print(token_pos.shape, token_pos.max())
        # print(batch_rep.contiguous().view(-1, dim).shape)
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B
        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def context_pooling(
        self, value_matrix, trigger_att, hidden_rep
    ):  # 文章中RLIG 和STCP的核心实现函数，基于value_matrix和trigger的注意力头得到对上下文和角色信息的关注度
        bsz = value_matrix.shape[0]
        rss = []
        for i in range(bsz):
            att = value_matrix[i] * trigger_att[i]
            att = att / (att.sum(1, keepdim=True) + 1e-5)  # 防止分母出现0
            rs = contract("ld,rl->rd", hidden_rep[i], att)
            rss.append(rs)
        return torch.stack(rss, dim=0)

    def forward(
        self,
        # ==== preprocess ====
        input_ids: Optional[List[int]] = None,
        spans: Optional[List[Tuple[int, int]]] = None,
        span_lens: Optional[List[int]] = None,
        trigger_index: Optional[List[int]] = None,
        # answer! for classification loss
        labels: Optional[List[List[int]]] = None,
        # answer! for boundary loss
        start_labels: Optional[List[List[int]]] = None,
        # answer! for boundary loss
        end_labels: Optional[List[List[int]]] = None,
        # ==== collator_fn ====
        label_masks: Optional[List[List[int]]] = None,
        info_dicts: Optional[List[List[TInfoDict]]] = None,
        attention_mask=None,
        # ==== None ====
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # TODO: single data는 문장 전체 텍스트 및 multi event를 포함하고 있어야 함.
        # TODO: 즉, Wikievents의 ontology를 참조하여 event와 role을 가져와야 함.

        # ---- Pretrained model ----
        # 1. pre-trained 모델에 대한 last_hidden_state와 attention을 가져옴
        # TODO: sliding window[zhou, 2021]을 통해 average encoding vector를 구해야함

        # [cls][ctx]context[ctx][evt1][evtt1]event type[evtt1][role1]role[role1][evt1]...[cls]

        # print(
        #     input_ids.shape,
        #     token_type_ids.shape,
        #     attention_mask.shape,
        #     head_mask.shape,
        #     inputs_embeds.shape,
        #     position_ids.shape,
        #     output_hidden_states.shape,
        #     return_dict.shape,
        # )
        # print(len(input_ids[0]), attention_mask.shape)
        # print("Min Input ID:", input_ids.min().item())
        # print("Max Input ID:", input_ids.max().item())
        # print("Vocab Size:", self.bert.config.vocab_size)
        # print(
        #     "Embedding Table Size:", self.bert.embeddings.word_embeddings.weight.size(0)
        # )
        last_hidden_state, attention = self.encode(
            input_ids,
            token_type_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            position_ids,
            output_hidden_states,
            return_dict,
        )

        # 2. bsz: batch size, seq_len: sequence length, hidsize: hidden size
        bsz, seq_len, hidsize = last_hidden_state.size()

        # 3. dropout을 적용
        last_hidden_state = self.dropout(last_hidden_state)

        # ---- MyModel -----

        # TODO: context embedding을 처리하는 부분은 그대로
        # TODO: role embeeding을 처리할 때에는 multi event임을 고려하여 role embedding을 가져와야 함

        role_emb = []
        # event_emb = bsz *
        event_emb = []
        loss = 0
        event_cnt_by_event = []
        # 1. role/event 토큰 정보 가져오기. 해당 정보들은 known!
        for i in range(bsz):
            event_emb.append([])
            role_emb.append([])

            event_cnt = len(info_dicts[i]["event_idx"])
            event_cnt_by_event.append(event_cnt)

            for j in range(event_cnt):
                event_emb[i].append(last_hidden_state[i][info_dicts[i]["event_idx"][j]])
                role_emb[i].append(last_hidden_state[i][info_dicts[i]["role_idxs"][j]])
            event_emb[i] = torch.stack(event_emb[i], dim=0)

        event_emb = torch.stack(event_emb, dim=0)
        # context feature(not depends on specific event)
        span_num = spans.size(1)

        global_feature = last_hidden_state

        global_att = attention.mean(1)

        final = global_feature

        final_att = global_att

        # for boundry loss

        start_feature = self.transform_start(final)
        end_feature = self.transform_end(final)

        len_state = self.len_embedding(span_lens)

        # print(spans.shape, start_feature.shape)
        # print maximum value from spans[:,:,0]
        # print(spans[:, :, 0].max())

        b_feature = self.select_rep(start_feature, spans[:, :, 0])

        e_feature = self.select_rep(end_feature, spans[:, :, 1])

        b_att = self.select_rep(final_att, spans[:, :, 0])

        e_att = self.select_rep(final_att, spans[:, :, 1])

        context = (
            torch.arange(seq_len)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(bsz, span_num, 1)
            .to(final)
        )

        context_mask = (context >= spans[:, :, 0:1]) & (context <= spans[:, :, 1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)

        context_feature = torch.bmm(context_mask, final)
        context_att = torch.bmm(context_mask, final_att)

        # by specific event
        # trigger_feature_by_event = torch.array()

        trigger_feature_by_event = []
        span_feature_by_event = []
        # [1, 1, 2, 4]
        for i in range(bsz):
            for j in range(event_cnt_by_event[i]):
                trigger_idx = trigger_index[i][j]

                # final: bsz * seq_len * dim
                # trigger_idx: bsz
                # trigger_feature: bsz * span_num * dim

                trigger_feature = (
                    self.select_single_token_rep(final, trigger_idx)
                    .unsqueeze(1)
                    .expand(-1, span_num, -1)
                )

                #
                trigger_att = (
                    self.select_single_token_rep(final_att, trigger_idx)
                    .unsqueeze(1)
                    .expand(-1, span_num, -1)
                )
                # context_pooling
                #
                b_rs = self.context_pooling(b_att, trigger_att, start_feature)
                #
                e_rs = self.context_pooling(e_att, trigger_att, end_feature)
                #
                context_rs = self.context_pooling(
                    context_att, trigger_att, global_feature
                )
                #
                b_feature_fin = torch.tanh(
                    self.begin_extractor(torch.cat((b_feature, b_rs), dim=-1))
                )
                #
                e_feature_fin = torch.tanh(
                    self.end_extractor(torch.cat((e_feature, e_rs), dim=-1))
                )
                #
                context_feature_fin = torch.tanh(
                    self.context_extractor(
                        torch.cat((context_feature, context_rs), dim=-1)
                    )
                )
                #
                span_feature = torch.cat(
                    (b_feature_fin, e_feature_fin, context_feature_fin), dim=-1
                )
                #
                span_feature = self.transform_span(span_feature)

                label = labels[:, j]
                start_label = start_labels
                end_label = end_labels

                if self.event_embedding is not None:

                    current_emb = event_emb[:, j, :]

                    logits = torch.cat(
                        (
                            span_feature,
                            trigger_feature,
                            torch.abs(span_feature - trigger_feature),
                            span_feature * trigger_feature,
                            len_state,
                            current_emb.unsqueeze(1).expand(-1, span_num, -1),
                        ),
                        dim=-1,
                    )
                else:
                    logits = torch.cat(
                        (
                            span_feature,
                            trigger_feature,
                            torch.abs(span_feature - trigger_feature),
                            span_feature * trigger_feature,
                            len_state,
                        ),
                        dim=-1,
                    )
                # P_{i:j}
                logits = self.classifier(logits)  # bsz * span_num * num_labels
                # label masks - bsz * evt_num * span_num * span_num
                # label_masks_expand - bsz * evt_num span_num * span_num
                label_masks_expand = (
                    label_masks[:, j].unsqueeze(1).repeat(1, span_num, 1)
                )
                # print(
                #     label_masks.shape, logits.shape, label_masks_expand.shape, span_num
                # )
                logits = logits.masked_fill(label_masks_expand == 0, -1e4)
                if label is not None:
                    # num_labels = max # of roles
                    focal_loss = MultiCEFocalLoss(self.num_labels)
                    print(logits.shape, label.shape)
                    loss += focal_loss(logits[label > -100], label[label > -100])

                # start/end boundary loss
                if self.lambda_boundary > 0:
                    # P_i^start
                    start_logits = self.start_classifier(start_feature)
                    # P_i^end
                    end_logits = self.end_classifier(end_feature)
                    if start_label is not None and end_label is not None:
                        loss_fct = CrossEntropyLoss(
                            weight=self.pos_loss_weight[:2].to(final)
                        )
                        loss += self.lambda_boundary * (
                            loss_fct(
                                start_logits.view(-1, 2),
                                start_label.contiguous().view(-1),
                            )
                            + loss_fct(
                                end_logits.view(-1, 2), end_label.contiguous().view(-1)
                            )
                        )

        loss /= len(event_cnt_by_event)
        print(loss)
        return {
            "loss": loss,
            "logits": logits,
            "spans": spans,
        }
