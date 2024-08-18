import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        # d_model is the dimension of the model
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        # create a matrix of size(seq_len,d_model)

        posm=torch.zeros(seq_len,d_model)

        # create a matx for representing the position of word in the sentence

        pos=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        # shape is (seq_len,1)
        # refer notes for the formula but yahi formula h Positional encoding ka

        divt=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        
        # apply the sin to even positions

        posm[:,0::2]=torch.sin(pos*divt)

        # apply the cos to odd positions
        posm[:,1::2]=torch.cos(pos*divt)

        # adding batch dim to tensor mtlb 2d se 3d krre
        posm=posm.unsqueeze(0) #ab dim hogyi(1,seq_len,d_model)

        # register to buffer to the model
        # what is buffer - when you want to keep a tensor not as a param but as a file the you register it in the buffer

        self.register_buffer("pe",posm)

    # adding the positional encoding to words in sentence

    def forword(self,x):
        x=x+(self.posm[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self,eps:float=10**-6 ) -> None:
        super().__init__()
        self.eps=eps
        # nn.Parameter makes the parameter learnable
        self.alpha=nn.Parameter(torch.ones(1)) #multiplied
        self.beta=nn.Parameter(torch.zeros(1)) #added

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)

        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std*self.eps)+self.beta
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int,d_ff:int,dropout:float) -> None:
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff) #W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) #W2 and B2

    def forward(self,x):
        # (batch,seq_len,d_model)-->(batch,seq_len,d_ff)-->(batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


                             

class MutiHeadAttentionBloack(nn.Module):

    def __init__(self, d_model:int,h:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h == 0, "d_model is not divisible by h"
        self.dk=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    # what is a static method-mtlb ki hum iss function ko kahi se bhi call krskte h without creating a instance
    # we can just say MultiHeadAttentionBlock.attention()
    def attention(Qp,Vp,Kp,mask,dropout:nn.Dropout):
        dk=Qp.shape[-1]
        attention_scores=(Qp @ Kp.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        # Yaha masking lagare taki jab softmax lge isme toh woh un values ko nullify krde

        attention_scores=attention_scores.softmax(dim=-1) #(Batch,h,seq_len,seqlen)

        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ Vp),attention_scores



    def forward(self,q,k,v,mask):
        Qp=self.w_q(q) # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        Kp=self.w_k(k) # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        Vp=self.w_v(v) # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        # Yaha prr humne multiply krra h

        # (batch,seqlen,d_model)-->(batch,seqlen,h,dk)-->(batch,h,seqlen,dk)
        # Yaha prr humne alag alag usme divide krdiya h
        Qp=Qp.view(Qp.shape[0],Qp.shape[1],self.h,self.dk).transpose(1,2)
        Vp=Vp.view(Vp.shape[0],Vp.shape[1],self.h,self.dk).transpose(1,2)
        Kp=Kp.view(Kp.shape[0],Kp.shape[1],self.h,self.dk).transpose(1,2)

        x,self.attention_scores=MutiHeadAttentionBloack.attention(Qp,Kp,Vp,mask,self.dropout)
        # (batch,h,seq_len,dk )-->(batch,seq_len,h,dk)-->(batch,seqlen,dmodel)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.dk)


        # (batch,seqlen,dmodel)-->(batch,seqlen,dmodel)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self,dropout:float)->None:

        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
# Yaha hum add norm layer lere h kyuki hum issi m apni previous layer bhejre
    def forward(self,x,sublayer):
# over hear we can see ki hum jo humare previous layer h sublayer usko hum usse add krdere h
        return x+self.dropout(sublayer(self.norm(x)))
    
    

# creating the encoder block

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block:MutiHeadAttentionBloack,feed_forward_block:FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        # isme hum mask isliye lagare taaki jo humari padding values h woh hath jye

        x=self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)

        return x

class Encoder(nn.Module):
# our encoder is made up of n encoder blocks toh hum woh lgayenge
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()

        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask):

        for layer in self.layers:
            x=layer(x,mask)

        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self,self_attention_block:MutiHeadAttentionBloack,cross_attention_block:MutiHeadAttentionBloack,feed_forward_block:FeedForwardBlock,dropout:float)->None:

        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_msk,tgt_msk):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,src_msk))
        x=self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,tgt_msk))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self,layers:nn.ModuleList)->None:
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_msk,tgt_msk):
        for layer in self.layers:
            x=layer(x,encoder_output,src_msk,tgt_msk)
        
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()

        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
# (batch,seqlen,d_model)-->(batch,seqlen,vocabsize)
        return torch.log_softmax(self.proj(x),dim=-1)

class Transformers(nn.Module):

    def __init__(self,encoder:Encoder,decoder:Decoder,input_embed:InputEmbeddings,output_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer)->None:
        self.encoder=encoder
        self.decoder=decoder
        self.input_embed=input_embed
        self.output_embed=output_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer
    
    def encode(self,src,src_mask):
        src=self.input_embed(src)
        src=self.src_pos(src)

        return self.encoder(src,src_mask)
    
    def decode(self,tgt,encoder_output,src_msk,tgt_msk):
        tgt=self.output_embed(tgt)
        tgt=self.tgt_pos(tgt)
        
        return self.decoder(tgt,encoder_output,src_msk,tgt_msk)
    
    def project(self,x):
        return self.projection_layer(x)
    
# Yaha tak humne sarre blocks jo humare use k banaliye h ab ek aisa function banana pdega jo inko sath m use krre hyperparameter k sath


def build_transformer(src_vocab:int,tgt_vocab:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff=2048):
    # Creatuing the embedding layers

    src_embed=InputEmbeddings(d_model,src_vocab)
    tgt_embed=InputEmbeddings(d_model,tgt_embed)

    # create the positional encoding layers
    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)

    # create the encoder blocks
    encoder_blocks=[]

    for _ in range(N):
        encoder_self_attention=MutiHeadAttentionBloack(d_model,h,dropout)
        feed_forward=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention,feed_forward,dropout)
        encoder_blocks.append(encoder_block)
    
    # create the decoder blocks

    decoder_blocks=[]

    for _ in range(N):
        decoder_self_attention=MutiHeadAttentionBloack(d_model,h,dropout)
        decoder_cross_attention=MutiHeadAttentionBloack(d_model,h,dropout)
        feed_forward=FeedForwardBlock(d_model,d_ff,dropout)

        decoder_block=DecoderBlock(decoder_self_attention,decoder_cross_attention,feed_forward,dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    # creating a projection layer

    projection_layer=ProjectionLayer(d_model,tgt_vocab)

    # Create the transformer

    tranformer=Transformers(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    # Initialize the parameters so they don't start with random values
    # we are using Xaviers algorithm here

    for p in tranformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return tranformer

