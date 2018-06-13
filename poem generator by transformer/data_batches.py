import numpy as np
def get_data(data_path):
    input_texts=[]
    target_texts=[]
    characters=set()
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                a=0
                input_text,target_text = line.strip().split(':')
                target_text = target_text.replace(' ','')
                input_text = input_text.replace(' ','')
                for i in [  '“','”','─','2','3', '6','9','C','D','F','M','c','d','g','h','j','l','o','p','q','z','ē','ń','…','□','、', '】','_','(','《','[']:
                    if i in input_text:
                        a=1
                        break
                    if i in target_text:
                        a=1
                        break
                if a==1:
                    continue
                if len(target_text) < 5 or len(target_text) > 80:
                    continue
                target_text = '\t' + target_text + '\n'
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    if char not in characters:
                        characters.add(char)
                for char in target_text:
                    if char not in characters:
                        characters.add(char)
            except Exception as e: 
                pass
    characters.add(' ')
    characters=sorted(list(characters))
    num_tokens=len(characters)
    max_encoder_seq_length=max([len(txt) for txt in input_texts])
    max_decoder_seq_length=max([len(txt) for txt in target_texts])

    print("Number of samples:",len(input_texts))
    print("Number of unique tokens:",num_tokens)
    print("Max sequence length for inputs:",max_encoder_seq_length)
    print("Max sequence length for outputs:",max_decoder_seq_length)

    token_index=dict([(char,i) for i,char in enumerate(characters)])
    int_str=dict([(i,char) for i,char in enumerate(token_index)])
    data_mix=sorted(zip(input_texts,target_texts),key=lambda x:len(x[1]))
    input_texts,target_texts=zip(*data_mix)

    encoder_input_texts_int=[[token_index[letter] for letter in line] for line in input_texts]
    decoder_input_texts_int=[[token_index[letter] for letter in line] for line in target_texts]
    decoder_target_texts_int=[line[1:] for line in decoder_input_texts_int]
    return encoder_input_texts_int,decoder_input_texts_int,decoder_target_texts_int,token_index,int_str,data_mix

#batch数据生成
def pad_sentence_batch(sentence_batch, pad_int,max_sentence):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    
    参数：
    - sentence batch
    - pad_int: ' '对应索引号
    '''
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    
def get_batches(batch_size,encoder_input_texts_int,decoder_input_texts_int,decoder_target_texts_int,decoder_pad_int, encoder_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(encoder_input_texts_int)//batch_size):
        start_i = batch_i * batch_size
        encoder_input_batch = encoder_input_texts_int[start_i:start_i + batch_size]
        decoder_input_batch = decoder_input_texts_int[start_i:start_i + batch_size]
        decoder_target_batch=decoder_target_texts_int[start_i:start_i + batch_size]
        encoder_length = max([len(sentence) for sentence in encoder_input_batch])
        decoder_length = max([len(sentence) for sentence in decoder_input_batch])
        # 补全序列
        Encoder_input_batch = np.array(pad_sentence_batch(encoder_input_batch, encoder_pad_int,encoder_length))
        Decoder_input_batch = np.array(pad_sentence_batch(decoder_input_batch, decoder_pad_int,decoder_length))
        Decoder_target_batch = np.array(pad_sentence_batch(decoder_target_batch, decoder_pad_int,decoder_length))

        # 获取目标的权重
        target_weights=np.zeros((batch_size,decoder_length),dtype='float32')
        for i ,line in enumerate(decoder_target_batch):
            for j,_ in enumerate(line):
                target_weights[i,j]=1
        yield Encoder_input_batch, Decoder_input_batch, Decoder_target_batch,target_weights