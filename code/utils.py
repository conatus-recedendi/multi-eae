MAX_NUM_EVENTS = 6
MAX_NUM_ARGUMENTS = 24
EXTERNAL_TOKENS = []
for i in range(MAX_NUM_EVENTS):
    EXTERNAL_TOKENS.append("<t-%d>" % i)
    EXTERNAL_TOKENS.append("</t-%d>" % i)
    EXTERNAL_TOKENS.append("<e-%d>" % i)
    EXTERNAL_TOKENS.append("</e-%d>" % i)

for i in range(MAX_NUM_ARGUMENTS):
    EXTERNAL_TOKENS.append("<r-%d>" % i)
    EXTERNAL_TOKENS.append("</r-%d>" % i)
