from typing import Optional, List, Tuple, TypedDict

from loss import MultiCEFocalLoss
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta import RobertaModel, RobertaConfig
from transformers.modeling_utils import PreTrainedModel
from opt_einsum import contract
from long_seq import process_long_input


class TInfoDict(TypedDict):
    event_idx: int
    event_ids: List[int]
    role_idxs: List[int]
    words_num: int


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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
        shift = (
            (torch.arange(B).unsqueeze(-1).expand(-1, num) * L)
            .contiguous()
            .view(-1)
            .to(batch_rep.device)
        )
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
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
        event_emb = []
        loss = 0
        # 1. role/event 토큰 정보 가져오기. 해당 정보들은 known!
        for i in range(bsz):
            for event in range(events):
                info_dict = info_dicts[i]
                pass

        for i in range(bsz):
            info_dict = info_dicts[i]
            # event idx의 위치의 토큰
            # bsz * # of event
            # event_emb.append(last_hidden_state[i][info_dict["#event_idx"]])
            event_emb.append(last_hidden_state[i][info_dict["event_idx"]])

            # role idx 위치의 토큰
            # bsz * # of event * # of role
            # role_emb.append(last_hidden_state[i][info_dict["#role_idxs"]])
            role_emb.append(last_hidden_state[i][info_dict["role_idxs"]])

            # 2. event_emb 변환
            event_emb = torch.stack(event_emb, dim=0)

            span_num = spans.size(1)

            # (batch_size, sequence_length, hidden_size)
            global_feature = last_hidden_state
            # (batch_size, num_heads, sequence_length, sequence_length)
            # -> (batch_size, sequence_length, sequence_length)
            global_att = attention.mean(1)
            final = global_feature
            final_att = global_att

            # for boundary loss
            # H^start

            # (batch_size, sequence_length, hidden_size)
            # h_i^start
            # info: span feature 가 아니다! for boundary
            start_feature = self.transform_start(final)
            # h_i^end
            # info: span feature 가 아니다! for boundary
            end_feature = self.transform_end(final)

            # TODO: trigger_index int -> List[int]
            # bsz * span_num * dim
            # h_t
            trigger_feature = (
                self.select_single_token_rep(final, trigger_index)
                .unsqueeze(1)
                .expand(-1, span_num, -1)
            )

            # TODO: trigger_index int -> List[int]
            # bsz * span_num * dim
            # A^t
            trigger_att = (
                self.select_single_token_rep(final_att, trigger_index)
                .unsqueeze(1)
                .expand(-1, span_num, -1)
            )

            # bsz * span_num * pos_size
            len_state = self.len_embedding(span_lens)

            # (batch_size, num_spans, dim)
            # h^start_i
            b_feature = self.select_rep(start_feature, spans[:, :, 0])
            # (batch_size, num_spans, dim)
            # h^end_i
            e_feature = self.select_rep(end_feature, spans[:, :, 1])
            # (batch_size, num_spans, dim)
            # b^s_{i:j}
            b_att = self.select_rep(final_att, spans[:, :, 0])
            # (batch_size, num_spans, dim)
            e_att = self.select_rep(final_att, spans[:, :, 1])

            # bsz * span_num * seq_len
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
            # c^s_{i:j}
            context_feature = torch.bmm(context_mask, final)
            # A^C_{i:j}
            context_att = torch.bmm(context_mask, final_att)  # bsz * span_num * hidsize

            """
          这里为了简便我们将STCP和RLIG的context_pooling操作合在一起。 global_feature包括上下文信息和角色信息。经过我们实验，分开计算STCP、
          RLIG和合并计算它们的效果差不多。
          kor: 여기서는 간편하게 하기 위해 STCP와 RLIG의 context_pooling 작업을 함께 수행합니다. 
          global_feature는 문맥 정보와 역할 정보를 포함합니다. 우리의 실험에 따르면, 
          STCP와 RLIG를 따로 계산하는 것과 함께 계산하는 것의 효과는 거의 비슷합니다.
          """
            # z^start_{i:j}
            b_rs = self.context_pooling(b_att, trigger_att, start_feature)

            # z^end_{i:j}
            e_rs = self.context_pooling(e_att, trigger_att, end_feature)

            # r^s_{i:j} ???
            context_rs = self.context_pooling(context_att, trigger_att, global_feature)

            # h^start_{i:j}
            b_feature_fin = torch.tanh(
                self.begin_extractor(torch.cat((b_feature, b_rs), dim=-1))
            )
            # h^end_{i:j}
            e_feature_fin = torch.tanh(
                self.end_extractor(torch.cat((e_feature, e_rs), dim=-1))
            )

            # s_{i:j}
            context_feature_fin = torch.tanh(
                self.context_extractor(torch.cat((context_feature, context_rs), dim=-1))
            )
            # 获取role embedding的表征
            span_feature = torch.cat(
                (b_feature_fin, e_feature_fin, context_feature_fin), dim=-1
            )
            # s^~_{i:j}
            span_feature = self.transform_span(span_feature)

            # I_{i:j}
            if self.event_embedding is not None:
                logits = torch.cat(
                    (
                        span_feature,
                        trigger_feature,
                        torch.abs(span_feature - trigger_feature),
                        span_feature * trigger_feature,
                        len_state,
                        event_emb.unsqueeze(1).expand(-1, span_num, -1),
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
            label_masks_expand = label_masks.unsqueeze(1).expand(-1, span_num, -1)
            logits = logits.masked_fill(label_masks_expand == 0, -1e4)
            if labels is not None:
                # num_labels = max # of roles
                focal_loss = MultiCEFocalLoss(self.num_labels)
                loss += focal_loss(logits[labels > -100], labels[labels > -100])

            # start/end boundary loss
            if self.lambda_boundary > 0:
                # P_i^start
                start_logits = self.start_classifier(start_feature)
                # P_i^end
                end_logits = self.end_classifier(end_feature)
                if start_labels is not None and end_labels is not None:
                    loss_fct = CrossEntropyLoss(
                        weight=self.pos_loss_weight[:2].to(final)
                    )
                    loss += self.lambda_boundary * (
                        loss_fct(
                            start_logits.view(-1, 2), start_labels.contiguous().view(-1)
                        )
                        + loss_fct(
                            end_logits.view(-1, 2), end_labels.contiguous().view(-1)
                        )
                    )
            # loss += ident_loss
        return {
            "loss": loss,
            "logits": logits,
            "spans": spans,
        }
