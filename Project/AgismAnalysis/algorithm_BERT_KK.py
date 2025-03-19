import numpy as np
import pandas as pd
import time
import datetime
from tqdm import tqdm, tqdm_pandas # execution time
tqdm.pandas()
from scipy.special import softmax
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
## Text Mining
from bertopic import BERTopic
from transformers import pipeline, AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers import AutoModel, BertModel, BertForSequenceClassification
from transformers import TFBertModel, TFBertForSequenceClassification
from transformers import BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt


def preprocessing_sentence_to_BERTinput(tokenizer, X_series, Y_series, 
                                        seq_len=128, batch_size=32, sampler=None):
    # BERT 입력 형식에 맞게 변환
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in X_series]
    
    # 전처리
    token_list = [tokenizer.encode_plus(sentence, max_length=seq_len,
                                        pad_to_max_length=True, truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True) for sentence in sentences]
    tokens = [token['input_ids'] for token in token_list]
    masks = [token['attention_mask'] for token in token_list]
    segments = [token['token_type_ids'] for token in token_list]
    
    # array 변환
    tokens = np.array(tokens)
    
    # tensor 변환
    tokens = torch.tensor(tokens)
    masks = torch.tensor(masks)
    segments = torch.tensor(segments)
    
    # Y 변환 및 정리
    if len(Y_series) != 0:
        targets = Y_series.values
        targets = np.array(targets)
        targets = torch.tensor(targets)
        data = TensorDataset(tokens, masks, targets)
    else:
        data = TensorDataset(tokens, masks)
    
    # pytorch dataloader 연결
    if sampler == None:
        dataloader = DataLoader(data, batch_size=batch_size)
    elif sampler == 'random':
        dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)
    elif sampler == 'sequential':
        dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)
        
    return dataloader


def modeling_BERTsentiment(device, model_name, train_dataloader, validation_dataloader,
                           num_labels=2, epochs=10, learning_rate=1.0e-5, l2_ratio=0,
                           early_stopping_patience=10,
                           save_location=None):
    # 하위 함수
    ## 정확도 계산 함수
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    ## 시간 표시 함수
    def format_time(elapsed):
        # 반올림
        elapsed_rounded = int(round((elapsed)))

        # hh:mm:ss으로 형태 변경
        return str(datetime.timedelta(seconds=elapsed_rounded))
      
    # 모델 로딩
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=learning_rate, # 학습률
                      eps=1e-8, weight_decay=l2_ratio) # 0으로 나누는 것을 방지하기 위한 epsilon 값, 과적합 방지 정규화 파라미터
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,    # 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    # 변수 설정
    ## early stopping
    patience_counter = 0
    best_loss = float('inf')
    best_model_state = None
    ## result
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    
    # 학습 및 검증
    model.zero_grad()    # Initialize gradient
    for epoch_i in range(0, epochs):
        # 학습
        print('\n======== Epoch {:} / {:} ======='.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_loss = 0
        model.train()
        train_accuracy = 0
        nb_train_steps = 0

        # Switch to training mode
        for step, batch in enumerate(tqdm(train_dataloader)):
            ## Put the batch on the GPU
            batch = tuple(t.to(device) for t in batch)
            ## Extract data from the batch
            b_input_ids, b_input_mask, b_labels = batch
            ## Perform Forward
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            ## Get the output
            loss = outputs[0]
            logits = outputs[1]
            total_loss += loss.item()
            
            # 정리
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_train_accuracy = flat_accuracy(logits, label_ids)
            train_accuracy += tmp_train_accuracy
            nb_train_steps += 1

            # Compute gradients by performing a backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # Gradient clipping
            optimizer.step()   # Update model's parameters using the gradients
            model.zero_grad()    # Initialize gradients
            scheduler.step()

        # 정리
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / nb_train_steps
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        print("Training Loss: {0:.2f}".format(avg_train_loss), 
              " Accuracy: {0:.2f}".format(avg_train_accuracy), 
              " Epoch Took: {:}".format(format_time(time.time() - t0)))
        
        # 검증
        t0 = time.time()    # Set start time
        model.eval()    # Switch to evaluation mode
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0
        ## PyTorch는 훈련 모드일 때는 loss와 logits를 반환하지만, 평가(evaluation) 모드에선 logits만 반환
        ## train 단계에서 CrossEntropyLoss를 활용해 loss를 별도로 계산
        loss_fn = CrossEntropyLoss()
        for batch in validation_dataloader:
            ## Put the batch on the GPU
            batch = tuple(t.to(device) for t in batch)
            ## Extract data from the batch
            b_input_ids, b_input_mask, b_labels = batch

            # Do not calculate gradients during validation
            with torch.no_grad():
                # Perform Forward
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            
            ## Get the output
            loss = loss_fn(logits, b_labels)
            eval_loss += loss.item()    # Ensure that loss is a scalar (0-dimensional tensor) and accumulate the loss

            # 정리
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        # 정리
        avg_valid_loss = eval_loss / len(validation_dataloader)
        avg_valid_accuracy = eval_accuracy / nb_eval_steps
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_accuracy)
        print("Validation Loss: {0:.2f}".format(avg_valid_loss),
              " Accuracy: {0:.2f}".format(avg_valid_accuracy), 
              " Epoch Took: {:}".format(format_time(time.time() - t0)))
        
        # 리스트에 결과 추가
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_accuracies.append(avg_train_accuracy)
        valid_accuracies.append(avg_valid_accuracy)
        
        # 조기종료
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print("Early Stopping Counter: {}/{}".format(patience_counter, early_stopping_patience))

        if patience_counter >= early_stopping_patience:
            print("Early Stopping triggered after {} epochs".format(epoch_i + 1))
            break
            
    # 결과 시각화
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(valid_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(valid_accuracies, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
            
    # 최적 모델 저장
    print("Learning Complete!")
    if best_model_state:         
        if save_location == None:
            save_location = os.path.join(os.getcwd(), 'Model', 'modeling_BERT_'+datetime.datetime.now().strftime("%Y%m%d")+'.pt')
            torch.save(best_model_state, save_location)
        else:
            torch.save(best_model_state, save_location)
            
    return model


def predict_sentiment_BERT(device, model, sentences, tokenizer,
                           seq_len=128, batch_size=32):
    # 문장을 입력 데이터로 변환
    test_dataloader = preprocessing_sentence_to_BERTinput(tokenizer, sentences, [],
                                                          seq_len=seq_len, batch_size=batch_size, sampler=None)

    # 예측
    model.eval()
    logits = []
    for batch_id, (tokens, masks) in enumerate(tqdm(test_dataloader)):
        b_input_ids = tokens.to(device)
        b_input_mask = masks.to(device)
        with torch.no_grad():    # 그래디언트 계산 안함
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    # Forward 수행

        # 출력
        logit = outputs[0].cpu().tolist()
        logits.extend(logit)
    logits = softmax(logits, axis=1)
    
    # 정리
    probability = pd.DataFrame(logits, columns=['Prob_Label'+str(i) for i in range(np.shape(logits)[1])])
    sentiment = pd.DataFrame(np.argmax(logits, axis=1), columns=['Sentiment'])
    
    return pd.concat([pd.Series(sentences).reset_index().iloc[:,1:], 
                      probability, sentiment], axis=1)

