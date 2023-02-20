from flask import Flask, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

def init():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )

    model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=True
    ).to(device='cuda', non_blocking=True)

    _ = model.eval()

def generate(prompt):

    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=512)
        generated = tokenizer.batch_decode(gen_tokens)[0]

    return generated

def form():
    return '<form method=post action=? id=form>Prompt: <textarea name=prompt cols=50 rows=8></textarea><button type=submit onClick="this.disabled=true; this.value=\'Processing...\'; document.getElementById(\'form\').submit();">Submit</button></form>'

@app.route('/', methods=['GET', 'POST'])
def index():
    input_form = form()
    print('prompt: ', request.form.get('prompt'))

    msg = ''
    prompt = request.form.get('prompt')

    if prompt is not None:
        msg = generate(prompt).replace('\n', '<br/>')
        prompt = prompt.replace('\n', '<br/>') + '<hr/>'

    return input_form + prompt  + msg

init()
print('Initialization is done, opening the service...')
app.run(host='0.0.0.0', port=8080)
