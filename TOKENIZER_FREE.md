# Tokenizer Independence

Tokenizer independence means domain bricks consume canonical ABI states and never token
IDs. It does not mean independently trained tokenized and byte-patch cores automatically
share coordinates.

The L6 gate requires paired-text evaluation of ABI drift plus domain/general PPL after
moving an unchanged brick between interfaces. Until that gate passes, the correct phrase
is **byte-patch ABI compatibility target**, not tokenizer-independent knowledge transfer.

The selected small-scale byte/byte-patch run now passes this bounded gate. A separate
2,048-piece byte-fallback BPE baseline reaches 2.4243 general BPB; the selected compact
byte-patch core reaches 2.4165 BPB with fewer parameters. This is evidence of parity on
one local corpus, not general tokenizer-free superiority.
