from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_PATH = Path(__file__).parent.parent.parent

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
