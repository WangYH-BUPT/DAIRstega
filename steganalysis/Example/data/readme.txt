_1、_2、_3：数字代表隐藏bit
mixbit_movie/tweet/news：检验混合隐藏比特后，不同语言风格的检测性能
mixstyle_1/2/3：检验混合语言风格后，不同隐藏比特的检测性能
不同语言风格不同隐藏比特检测时，句长≤25,cover=stego=2000


混合检测时句长不限制，
mixbit_movie/tweet/news的cover样本数为2000，每种隐藏bit的stego样本数各400
cover=1(style)*2000
stego=1(style)*5(bit)*400=2000

mixstyle_1/2/3的样本数均为3000，每种媒体style的样本各1000
cover=3(style)*1000=3000
stego=3(style)*1(bit)*1000

mixall的样本数为3种媒体style，5种隐藏bit,其中每种bit各200
cover=3(style)*1000=3000
stego=3*5*200=3000
