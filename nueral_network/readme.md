# 인공신경망 개요
## foward propagation
5개의 입력 $`x_i(1\leq i\leq 5 , 스칼라)`$, 은닉계층이 3개, 출력계층 뉴런 3개(3개의 계층 각각은 3,4,3개의 뉴런) $`y_i(1\leq i\leq 2, 스칼라)`$가 존재할 때, 진입간선으로 들어오는 입력에 대한 가중치 벡터  $`w_i(1\leq i\leq 3+4+3+2,벡터)`$를 가질 것이다. 이 신경망이 4개의 개념으로 classification 한다면, 각각의 출력계층이 2진 classification 해야한다. 이를 위해 출력계층의 함수값을 logistic함수($`logistic(z)=\frac{1}{1+e^{-z}}`$)에 합성시켜야 한다.
그리고 각 뉴런들이 중복된 기능을 하지 않도록 비선형($`f(x)+f(y)\neq f(x+y)`$)화 시키기 위해 hyperbolic tangent($`h(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}`$) 시켜야 한다.
위 의 내용들이 반영된 신경망의 출력값을 수식으로 표현하면 아래와 같다.

$`y_1=logistic(w_{11}\cdot (\displaystyle\sum_{l=8}^{10}{w_l}\cdot h(\displaystyle\sum_{j=4}^{7}{w_j}\cdot h(\displaystyle\sum_{k=1}^{3}{w_k}\cdot X))))`$

$`y_2=logistic(w_{12}\cdot (\displaystyle\sum_{l=8}^{10}{w_l}\cdot h(\displaystyle\sum_{j=4}^{7}{w_j}\cdot h(\displaystyle\sum_{k=1}^{3}{w_k}\cdot X))))`$ 

위 식에서 $`X`$는, 입력 스칼라 $`x_1, ... ,x_5`$으로 이루어진 벡터

## error function

### regression을 위해 output 뉴런의 활성화 함수가 identification 함수인 경우

4개의 분류 객체를 $`t_1,t_2,t_3,t_4`$라고 할때, $`t_i`$는 입력 벡터 $`X`$와 가중치 벡터 $`\mathbf w`$에 의한 정규분포라고 할  수 있으며, training set 중 하나의 element $`X_1`$를 모집단($`t_i`$)의 표본이자 표본평균이라고 할 수 있다. 그리고 training dataset이 많다면 모집단은 정규분포를 따른다.

예를 들어 설명하면 그림에서 사자,호랑이,새, 말로 분류할 때, 각각의 샘플그림에 분류객체들로 라벨링되어 있을 것이다. 샘플그림의 수가 많다면, 그림들은 각각의 분류객체를 대상으로 정규분포를 따를 것이다.
이 때 라벨링되지 않은 새로운 그림을 신경망에 넣으면, 임의의 가중치 상태에서 $`y(x,w)`$를 얻게 되는데, 이를 모집단에서의 표본으로 볼 수 있으며, 표본평균으로도 볼 수 있다(표본의 갯수가 1인).

따라서 신경망의 출력값은 표본평균의 정규분포를 따르기 때문에, 정규분포의 확률밀도함수를 사용하여  샘플의 feed forward 값(표본 평균)과 샘플의 라벨(모평균)의 유사도를 확인할 수 있다.

유사도(likelihood)는 정규분포의 확률밀도함수(likelihood)로 확인할 수 있으며, 확률밀도 함수는 아래와 같다.

$`f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}`$

training 단계에서는, 각 epoch 마다 동일한 sample에 대하여 유사도(확률밀도함수의 함수값)가 최대가 되도록 가중치를 업데이트 한다. 즉 가중치에 종속적인 f(x)의 최대값을 찾으면 되는데, 계산을 쉽게하기 위해 log를 취하면 유사도 함수는 아래와 같아진다.
$`L(x)=-\frac 12ln(2\pi)-\frac 12ln(\sigma^2)-\frac{1}{2}(\frac{x-\mu}{\sigma})^2`$

그런데 target과 입력변수는 갖고 x는 w에 의해 변하기 때문에 최대값을 구하는데 불필요한 수식들을 정리하면 아래와 같아진다.

$`L^*(x)=-\frac 12(x-\mu)^2`$ 여기서
-부호를 취하고 모든 target이 한번에 반영되도록 해주고 종속관계를 정리하면 아래와 같은 erro fucntion 이 된다.

$`E(w)=\frac 12\displaystyle\sum_{n=1}^{N}{(y(x_n,w)-t_n)}`$

유사도 함수는 최대값을 구하였지만, 유사도함수에 음의부호를 붙인 error 함수는 최소값을 찾아야 가중치가 target에 최적화가 된다.

regression을 위한 신경망의 error function을 최적화 하기위해 error function의 미분을 구해야 한다.  canonical link function을 이용하여 $`\frac{\partial E}{\partial a_k}=y_k-t_k`$임을 알 수 있는데, 이 사실을 이용하여 error fucntion을 미분하는데 참고하면 용이하다.


### categorical classification을 위해 output 뉴런의 활성화 함수가 logistic인 경우
이제 single target variable 5에 대한 2진 classification을 고려해 보자. t의 값이 1이면 class 1을 의미하고 t의 값이 0이면 class 2라고 할때, 활성화 함수로 logistic sigmoid $`y(x,y)=\frac{1}{1+e^{-a}}`$ 를 사용한 t에 의한 베르누이 분포로 생각 할 수 있다.<center>$`p(t|x,w)=y(x,w)^t\{1-y(x,w)\}^{1-t}`$</center>
따라서 2진 classification의 활성화 함수에 대한 likelihood function은 위와 같기 때문에, 위 식에 negtive logarithm을 취하게 되면 우리는 2진 classification 활성화 함수에 대한 error function을 얻을 수 있게 된다.<center>$`E(w)=-\displaystyle\sum_{n=1}^{N}{\{t_n\ln y_n+(1-t_n)\ln {(1-y_n)}\}}`$</center>
위 식에서 $`y_n`$은 $`y(x_n,w)`$를 의미하며 linear regression과 달리 noise precision $`\beta`$ 같은것이 없다. 왜냐하면 target value들이 정확히 labelling 되었다고 가정되기 때문이다.



#### K개의 이진 분류를 수행하는 경우
output unit의 logistic sigmoid 활성화 함수가 K개인 경우, 신경망의 label 변수 t는 $`t_k\in{0,1}, where k=1,..,K`$로 표현 된다.  class label 들이 서로 독립이면 target 변수에 대한 조건부확률은 아래와 같다.<center>$`p(t|x,w)=\displaystyle\prod_{k=1}^{K}{y_k(x,w)^{t_k}[1-y_k(x,w)]^{1-t_k}}`$</center>
 likelihood 함수이고, 여기에 negative logarithm을 취하면 아래와 같은 error function을 얻을 수 있다.
$`E(w)=-\displaystyle\sum_{n=1}^{N}{\displaystyle\sum_{k=1}^{K}{t_{nk}\ln{y_nk}+(1-t_{nk})\ln{(1-y_{nk})}}}`$

regression과 마찬가지로 특정 output unit에 대한 activation으로 error function을 미분하면 $`\frac{\partial E}{\partial a_k}=y_k-t_k`$ 형태를 갖는다.

### softmax
마지막으로 각 입력에 상호 배타적인 class K중 하나가 할당되는 multiclass classification 문제를 고려해 보자. 이진 타겟 변수들 $`t_k\in{0,1}`$는 class를 나타내는 1/K 코딩 스킴을 갖고 있으며, 신경망의 output $`y_k(x,w)`$는 $`p(t_k=1|x)`$로 해석될 수 있다.  이 output의 activation함수로 아래의 softmax가 적절한 역할을 할수 있다.<center>$`y_k(x,w)=\frac{e^{a_k(x,w)}}{\displaystyle\sum_j{e^{a_j(x,w)}}}`$</center>
위 식은 다음을 만족한다 : $`0\le y_k\le1`$, $`\sum_k{y_k}=1`$

그리고 이 softmax activation은 아래와 같은 error function으로 유도된다.<center>$`E(w)=-\displaystyle\sum_{n=1}^{N}{\displaystyle\sum_{k=1}^{K}{t_{kn}\ln{y_k(x_n,w)}}}`$</center>

softmax 또한 특정 output unit에 대한 activation으로 error function을 미분하게되면  $`\frac{\partial E}{\partial a_k}=y_k-t_k`$와 비슷한 형태가 된다.

## back propagation

역전파는 error function의 최소값을 찾는 과정으로 newton method, quasi newton, BFGS 알고리즘 같은 최적화 알고리즘을 사용한다. 이 알고리즘들은 공통적으로 함수의 1계도함수를 요구하기 때문에 우리는 error 함수의 1계도함수를 구해야 한다.

error 함수는 가중치 벡터 $`w_i\in W`$에 종속인 다변수 함수이므로, 각각의 가중치들 로 편미분된 gradient(Jacobian matrix)를 구해야 한다.

다행히도 output unit의 error function의 1계도함수에서 연쇄적으로 모든 가중치에 대한 편미분을 구할 수 있으며, ouput unit의 error function의 1계도함수는 canonical link function의 성질을 이용하여 쉽게 구하 수 있다. 1계도함수와 canonical link function의 관계는 좀더 연구하여 자세히 다루도록 하겠습니다.

output unit에서 각각의 가중치에 대한 편미분을 구하는 방법은 아래와 같다.

1. 신경망에  입력 벡터 $`x_n`$을 적용시켜 forward propgate를 진행시킨다.<center>$`a_j=\displaystyle\sum_i{w_{ji}z_i}`$</center><center>$`z_j=h(a_j)`$</center>
2.  $\delta_k=y_k-t_k$ 를 사용하여 모든 output unit에 대한 $\delta_k$를 구하라.
3. $\delta_j=h'(a_j)\displaystyle\sum_k{w_{kj}\delta_k}$를 사용하여 신경망의 모든 은닉 계층의 $\delta$를 구하라. 위 식에서 $h'(a_j)$는 은닉 계층의 활성화 함수의 미분을 의미한다. <center>$h'(a_j)=tanh'(a_j)\\=sech^2(a_j)\\=1-tanh(a_j)^2\\=1-h(a_j)^2$<center>

4. $\frac{\partial E_n}{\partial w_{ji}}=\delta_j z_i$를 사용하여 필요한 미분계수들을 구하라. batch method인 경우 



2번 과정에서 $`\delta`$는 regression을 위해 output activation함수가 identity 함수일 경우에 해당한다. output activation 함수가 logistic sigmoidal일 경우 <center>$\frac{\partial y_k}{\partial a_j}=\delta_{kj}\sigma'(a_j)$이고</center>
output activation이 softmax 일 경우<center>
$\frac{\partial_{yk}}{\partial_a}=\delta_k y_k - y_k y_j$가 된다.</center>
