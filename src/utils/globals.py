from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_PATH = Path(__file__).parent.parent.parent

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
