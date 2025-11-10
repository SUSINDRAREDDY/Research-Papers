import torch
import sys
from transformer import Transformer, make_src_mask, make_tgt_mask
from data_loader import src_vocab, tgt_vocab
import spacy

spacy_eng = spacy.load("en_core_web_sm")

SRC_PAD_IDX = src_vocab["<pad>"]
TGT_PAD_IDX = tgt_vocab["<pad>"]
TGT_SOS_IDX = tgt_vocab["<sos>"]
TGT_EOS_IDX = tgt_vocab["<eos>"]
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 256
DROPOUT = 0.1

device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(len(src_vocab), len(tgt_vocab),
                    d_model=D_MODEL, num_layers=NUM_LAYERS,
                    num_heads=NUM_HEADS, hidden_layers=D_FF, dropout=DROPOUT).to(device)

model.load_state_dict(torch.load("transformer_safe.pth", map_location=device))
model.eval()

idx_to_tgt = {idx: word for word, idx in tgt_vocab.items()}

def translate(text, max_len=50):
    tokens = [t.text.lower() for t in spacy_eng.tokenizer(text)]
    src_ids = [src_vocab.get(t, src_vocab["<unk>"]) for t in tokens]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    src_mask = make_src_mask(src_tensor, SRC_PAD_IDX).to(device)
    with torch.no_grad():
        enc_out = model.encoder(src_tensor, src_mask)
    
    tgt_ids = [TGT_SOS_IDX]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)
        tgt_mask = make_tgt_mask(tgt_tensor, TGT_PAD_IDX).to(device)
        
        with torch.no_grad():
            logits = model.decoder(tgt_tensor, enc_out, src_mask, tgt_mask)
            next_token = torch.argmax(logits[0, -1, :]).item()
        
        tgt_ids.append(next_token)
        if next_token == TGT_EOS_IDX:
            break
    
    words = []
    for idx in tgt_ids[1:]:
        word = idx_to_tgt[idx]
        if word in ["<eos>", "<pad>"]:
            break
        words.append(word)
    
    return " ".join(words)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        output = translate(input_text)
        print(output)
    else:
        print("Enter English text (Ctrl+C to exit):")
        while True:
            try:
                text = input("> ")
                if text.strip():
                    output = translate(text)
                    print(output)
            except KeyboardInterrupt:
                break

