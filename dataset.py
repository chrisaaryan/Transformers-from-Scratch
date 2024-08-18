import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len)->None:

        super().__init__()
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.seq_len=seq_len

        self.sos_token=torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])],dtype=torch.int64)
        self.eos_token=torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])],dtype=torch.int64)
        self.pad_token=torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index:any) -> any:
        src_tgt_pair=self.ds[index]
        # getting the src and tgt text
        src_text=src_tgt_pair['translation'][self.src_lang]
        tgt_text=src_tgt_pair['translation'][self.tgt_lang]

        # tokenizing them
        # text-->tokens-->input_ids
        encode_input_tokens=self.tokenizer_src.encode(src_text).ids
        # encode se tokens krre and ids se input id

        decode_input_tokens=self.tokenizer_tgt.encode(tgt_text).ids

        # padding to reach seq_len
        enc_num_padding_tokens=self.seq_len-len(encode_input_tokens)-2
        # why -2 because we need to add SOS and EOS

        dec_num_padding_tokens=self.seq_len-len(decode_input_tokens)-1
        # decoder side we only add the SOS token that's why

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError("sentence too long")

        # making the encoder decoder input and label

        encoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(encode_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input

        decoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(decode_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )
        
        # Add the EOS label (what we expect as output from the output])

        label=torch.cat(
            [
                torch.tensor(decode_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        # checking the size for the input and label

        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            "encoder_input":encoder_input, #(seq_len)
            "decoder_input":decoder_input, #(seq_len)
            # encoder mask and decoder mask we are making to ensure that our pad_token don't take part in self attention
            "encoder_mask":(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),#(1,1,seq_len)
            "decoder_mask":(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1,seq_len,seq_len)
            "label":label,#(seq_len)
            "src_text":src_text,
            "tgt_text":tgt_text

        }
    
def causal_mask(size):
    # gives us value above the diagnol of the matrix and rest will be zero
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    
    # this statement will return all the values jo zero hogyi h
    return mask==0
