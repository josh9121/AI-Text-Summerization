from django.shortcuts import render
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, BertForQuestionAnswering, BertTokenizer
import torch
summarizer = pipeline("summarization")
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def summarize_and_answer(request):
    context = {}
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        question = request.POST.get('question')
        summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        inputs = qa_tokenizer.encode_plus(question, input_text, add_special_tokens=True, return_tensors='pt')
        input_ids = inputs['input_ids'].tolist()[0]
        with torch.no_grad():
            outputs = qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        context = {
            'summary': summary,
            'answer': answer.title(),
            'input_text': input_text,
            'question': question
        }
    return render(request, 'index.html', context)

