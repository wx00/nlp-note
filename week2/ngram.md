## 语言模型（language model）

序列： 
&emsp;&emsp; ${\bf{w}} = (w_1w_2\cdots w_k)$
&emsp;
序列的概率`(chain rule)`: 
&emsp;&emsp; $p({\bf{w}}) = p(w_1)p(w_2|w_1)p(w_3|w_1w_2)\cdots p(w_k|w_1\cdots w_{k-1})$
&emsp;
马尔可夫假设`(n-gram markov assumption)`:
&emsp;&emsp; $p(w_i|w_1\cdots w_{i-1}) = p(w_i|w_{i-n+1}\cdots w_{i-1})$
&emsp;
二元马尔可夫假设`(2-gram)`:
&emsp;&emsp; $p(w) = p(w_1)p(w_2|w_1)p(w_3|w_2)\cdots p(w_k|w_{k-1})$
&emsp;

### 两个问题
-   规范化首个单词的概率：增加虚拟前缀 
    $p(w_1) \to p(w_1| {\rm{start}})$
-   使所有序列的概率之和为 1：增加虚拟后缀
    $p(w_k|w_{k-1}) \to p(w_k|w_{k-1})p({\rm{end}}|w_k)$

增加前后缀： ${\bf{w}} \to (w_0w_1\cdots w_kw_{k+1})$

&emsp;
二元模型:
$$ 
p({\bf{w}}) = \prod_{i=1}^{k+1} p(w_i|w_{i-1}) \\
p(w_i|w_{i-1}) = \frac{C(w_{i-1}w_i)}{C(w_{i-1})}$$
N元模型:
$$ 
p({\bf{w}}) = \prod_{i=1}^{k+1} p(w_i|w^{i-1}_{i-n+1}) \\
p(w_i|w^{i-1}_{i-n+1}) = \frac{C(w_{i-n+1}\cdots w_{i-1}w_i)}{C(w_{i-n+1}\cdots w_{i-1})}$$
N元模型的对数似然:$$ 
\log p({\bf{w_{\rm{train}}}}) = \sum_{i=1}^{N+1}\log p(w_i|w^{i-1}_{i-n+1})$$ 

&emsp;
评价模型的指标——困惑度`(perlexity)`:$$ 
p({\bf{w_{\rm{test}}}})^{-\frac{1}{N}} = \frac{1}{\sqrt[N]{p({\bf{w}})}}$$ 

&emsp;
未登录词处理——平滑处理`(smoothing)`：解决概率为 0 的问题
原理：概率重分布——将部分高频（已知）词的概率转移到低频（未知）词上

- 拉普拉斯平滑$$ 
\hat{p}(w_i|w^{i-1}_{i-n+1}) = \frac{c(w^{i}_{i-n+1}) + 1}{c(w^{i-1}_{i-n+1}) + V} \\
\hat{p}(w_i|w^{i-1}_{i-n+1}) = \frac{c(w^{i}_{i-n+1}) + k}{c(w^{i-1}_{i-n+1}) + Vk}
$$ 

- Katz Backoff$$ 
\hat{p}(w_i|w^{i-1}_{i-n+1}) = \begin{cases}
\tilde{p}(w_i|w^{i-1}_{i-n+1}) & C(w^{i}_{i-n+1})>0 \\
\alpha(w^{i-1}_{i-n+1})\hat{p}(w_i|w^{i-1}_{i-n+2}) & {\rm{otherwise}}
\end{cases}
$$ 

- 插值平滑$$ 
\hat{p}(w_i|w_{i-2}w_{i-1}) = \lambda_1p(w_i|w_{i-2}w_{i-1})+\lambda_2p(w_i|w_{i-1})+\lambda_3p(w_i) \\ {\bf{s.t.}} \\
\lambda_1+\lambda_2+\lambda_3=1
$$ 

- 绝对折扣$$ 
\hat{p}(w_i|w_{i-2}w_{i-1}) = \frac{C(w_{i-1}w_i)-d}{C(w_{i-1})} + \lambda(w_{i-1})p(w_i) 
$$ 

- Kneser-Ney$$ 
\hat{p}(w_i) \propto |\{x: C(xw_i)>0\}|
$$ 






&emsp;
&emsp;