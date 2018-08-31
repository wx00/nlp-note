"""
    Chain Rule
    p(w) = p(w1)p(w2|w1)...p(w_k|w1...w_k-1)

    Markov assumption: on care about the nearest n word before i
    p(w_i|w1...w_i-1)=p(w_i|w_i-(n+1)...w_i-1))

    Bigram language model: n = 2
    p(w) = p(w1)p(w2|w1)...p(w_k|w_k-1)

    Two problem:
    1. some word has never present at the start in training corpus, so it`s probability at the beginning p(w) will be zero
        use fake 'start' token to fix it
    2. sum(probability) of word sequence will not be 1
        add fake 'end' token to fix it

    Model:
        p(w) = \prod_{i=1}^{k+1} p(w_i|w_{i-1})

    Estimate Prob:
        p(w_i|w_{i-1}) = \frac_{c(w_{i-1}w_i)}{\sum_{w_i}{w_{i-1}w_i}}
                        = \frac_{c(w_{i-1}w_i)}{c(w_{i-1})}

    n-gram model:
        p(w) = \prod_{i=1}^{k+1} p(w_i|w^{i-1}_{i-n+1})

    5-gram should be good enough

    Likelihood:
        p(w) = \prod_{i=1}^{k+1} p(w_i|w^{i-1}_{i-n+1})

    Perplexity:
        p(w)^{-\frac{1}{k}} = \frac{1}{\sqrt[N]{p(w)}}

    Smoothing: process unknown word
        Laplacian :
            add one: \hat{p}(w_i) = c(w_i) + 1 / c(w_{i-1}) + V
            add k: \hat{p}(w_i) = c(w_i) + k / c(w_{i-1}) + V*k

        Katz backoff :
            \hat{p}(w_i) =
                p(w_i)
                \alpha() p()

        Interpolation:
            



"""