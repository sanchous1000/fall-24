# Решающее дерево

Решающее дерево — это важный инструмент в машинном обучении, который позволяет принимать решения на основе данных. В процессе создания дерева важно выбрать правильный критерий для разделения вершин. Рассмотрим два критерия: критерий Донского и многоклассовый энтропийный критерий.

```python
def explicit_predict(feature):                                                                                     
    if feature[1] == 'male':                                                                                       
        if feature[5] <= 15.5:                                                                                     
            return 0                                                                                               
        else:                                                                                                      
            if feature[0] <= 1.0:                                                                                  
                if feature[2] <= 42.0:                                                                             
                    return 1                                                                                       
                else:                                                                                              
                    return 0                                                                                       
            else:                                                                                                  
                if feature[2] <= 9.0:                                                                              
                    if feature[3] <= 1.0:                                                                          
                        return 1                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
                else:                                                                                              
                    return 0                                                                                       
    else:                                                                                                          
        return 1   
```

## Критерии для разделения вершин

### Критерий Донского

Критерий Донского (или метод, разработанный Владимиром Донским) относится к категории адаптивных методов обучения решающих деревьев. Он направлен на минимизацию распределенного риска и позволяет эффективно обрабатывать разреженные данные. Основной принцип заключается в оптимизации градиентного дерева путем оценки риска для различных возможных разбиений и выбора оптимального.

Применение этого критерия требует продвинутых теоретических знаний в области статистического обучения и просто так его не применяют без соответствующего теоретического обоснования и подходящих данных.
Критерий Донского основан на следующей формуле информативности предиката $\beta$ на множестве объектов $U$:

$$
I(\beta, U) = \frac{|U_0| \cdot |U_1|}{|U|^2} \sum_{y \in Y} \left|\frac{|U_0^y|}{|U_0|} - \frac{|U_1^y|}{|U_1|}\right|
$$

где:
- $U_0$ и $U_1$ — подмножества объектов, на которых предикат $\beta$ принимает значения 0 и 1 соответственно
- $U_0^y$ и $U_1^y$ — подмножества объектов класса $y$ в $U_0$ и $U_1$ соответственно
- $|U|$ — мощность множества $U$ (количество объектов)

При реализации этого критерия следует:

1. Для каждого признака и порога разбиения:
   - Разделить выборку на две части ($U_0$ и $U_1$)
   - Для каждого класса посчитать доли объектов в обеих частях
   - Вычислить модуль разности этих долей
   - Просуммировать по всем классам
   - Умножить на нормировочный множитель $\frac{|U_0| \cdot |U_1|}{|U|^2}$

2. Выбрать признак и порог с максимальным значением критерия

### Многоклассовый энтропийный критерий

Многоклассовая энтропия, также известная как критерий энтропии или энтропийный индекс, основан на концепции информации. Он измеряет степень неопределенности в выборке и использует понятие энтропии из теории информации. Энтропия $H$ определяется как:

$$
H(U) = -\sum_{y \in Y} p_y \log_2(p_y)
$$

где:
- $p_y$ — доля объектов класса $y$ в множестве $U$
- $Y$ — множество всех классов

Информативность предиката $\beta$ на множестве объектов $U$ определяется как уменьшение энтропии после разбиения:

$$
I(\beta, U) = H(U) - \frac{|U_0|}{|U|}H(U_0) - \frac{|U_1|}{|U|}H(U_1)
$$

где:
- $H(U)$ — энтропия до разбиения
- $H(U_0)$ и $H(U_1)$ — энтропии подмножеств после разбиения
- $|U_0|$ и $|U_1|$ — количество объектов в подмножествах
- $|U|$ — общее количество объектов

При реализации этого критерия следует:

1. Рассчитать начальную энтропию множества:
   - Посчитать частоты каждого класса
   - Вычислить энтропию по формуле выше

2. Для каждого признака и порога разбиения:
   - Разделить выборку на две части ($U_0$ и $U_1$)
   - Рассчитать энтропию для каждой части
   - Вычислить взвешенную сумму энтропий
   - Вычесть из начальной энтропии

3. Выбрать признак и порог с максимальным значением информационного выигрыша

Этот критерий особенно эффективен, когда:
- Данные содержат множество классов
- Распределение классов неравномерно
- Требуется интерпретируемая модель с понятной метрикой качества разбиения

### Различия между критериями

1. **Цели и природа:** Критерий Донского — это более продвинутый метод, ориентированный на адаптацию и минимизацию распределенного риска для продвинутых многомерных данных. Энтропийный критерий — основной, интуитивно понятный и часто используется в стандартных задачах классификации для уменьшения неопределенности.

2. **Применимость:** Энтропийный критерий проще в реализации и чаще используется в популярных библиотеке, таких как Scikit-learn, особенно для задач с множественными классами. Критерия Донского требует уникального подхода к каждой задаче и часто применяется при наличии специфических требований к модели.

### Пример реализации на Python с использованием numpy

```python
class DecisionTreeID3:
    def __init__(
        self,
        criterion: Literal["entropy", "donskoy"] = "entropy",
        min_samples_split: int = 2,
        max_depth: int = 5,
    ):
        """
        Initialize ID3 decision tree classifier

        Parameters:
        -----------
        criterion : str, default='entropy'
            The function to measure the quality of a split.
            Supported criteria are 'entropy' for information gain with entropy
            and 'donskoy' for Donskoy's criterion
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node
        max_depth : int, default=5
            The maximum depth of the tree
        """
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
        self._explicit_predict = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
    ):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).
        feature_types : list[Literal["categorical", "numeric"]]
            The types of features.
        """
        self.tree = self._build_tree(X, y, feature_types)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
        depth: int = 0,
    ) -> Node:
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        # Base cases
        if n_classes == 1:
            return LeafNode(class_=y[0])
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return LeafNode(class_=np.argmax(np.bincount(y)))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y, feature_types)

        # Split the data
        match best_threshold:
            case float():
                # Numeric feature
                left_mask = X[:, best_feature] <= best_threshold
                right_mask = ~left_mask
            case _:
                # Categorical feature
                left_mask = X[:, best_feature] == best_threshold
                right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(
            X[left_mask], y[left_mask], feature_types, depth + 1
        )
        right_subtree = self._build_tree(
            X[right_mask], y[right_mask], feature_types, depth + 1
        )

        return ParentNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
    ) -> tuple[int, float | str]:
        _, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            if feature_types[feature] == "categorical":
                unique_values = list(set(X[:, feature]))
                # Try each category as a binary split
                for category in unique_values:
                    gain = self._information_gain(X[:, feature], y, category)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = category
            else:
                unique_values = np.unique(X[:, feature])
                # Numeric feature - use threshold splits
                for threshold in unique_values:
                    gain = self._information_gain(X[:, feature], y, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(
        self,
        X_column: np.ndarray,
        y: np.ndarray,
        threshold: float | Any,
    ) -> float:
        # Split the data based on threshold
        match threshold:
            case float() if np.isnan(threshold):
                # If threshold is NaN, return 0 gain
                return 0.0
            case float():
                try:
                    left_mask = X_column <= threshold
                    right_mask = ~left_mask
                except TypeError:
                    raise ValueError(
                        f"Threshold must be a number for numeric features, got {threshold}: {type(threshold)}"
                    )
            case _:
                left_mask = X_column == threshold
                right_mask = ~left_mask

        # Get the child node samples
        left_y = y[left_mask]
        right_y = y[right_mask]

        if self.criterion == "entropy":
            # Calculate parent entropy
            parent_entropy = self._entropy(y)

            # Calculate weights of splits
            n_samples = len(y)
            w_left = len(left_y) / n_samples
            w_right = len(right_y) / n_samples

            # Calculate weighted child entropy
            child_entropy = w_left * self._entropy(left_y) + w_right * self._entropy(
                right_y
            )
            return parent_entropy - child_entropy

        else:  # donskoy criterion
            n_total = len(y)
            n_left = len(left_y)
            n_right = len(right_y)

            # Normalization factor
            norm_factor = (n_left * n_right) / (n_total**2)

            # Calculate class proportions difference
            total_diff = 0
            for class_label in np.unique(y):
                left_prop = np.sum(left_y == class_label) / n_left if n_left > 0 else 0
                right_prop = (
                    np.sum(right_y == class_label) / n_right if n_right > 0 else 0
                )
                total_diff += abs(left_prop - right_prop)

            return norm_factor * total_diff

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a node."""
        # Handle empty arrays
        if len(y) == 0:
            return 0.0

        # Calculate probabilities of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # Calculate entropy using formula: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X."""
        return np.array([self._predict_row(row) for row in X])

    def _predict_row(self, row: np.ndarray) -> int:
        """Predict class for a single row."""
        node = self.tree
        while True:
            match node:
                case LeafNode(class_=class_):
                    return class_
                case ParentNode(
                    threshold=threshold,
                    feature=feature,
                    left=left,
                    right=right,
                ):
                    match threshold:
                        case float():
                            go_left = row[feature] <= threshold
                        case _:
                            go_left = row[feature] == threshold

                    if go_left:
                        node = left
                    else:
                        node = right
                case _:
                    raise ValueError(f"Invalid node: {node}")
```


## Редукция дерева

Алгоритм редукции (или обрезания) дерева — это техника, используемая для уменьшения сложности решающего дерева после его создания. Основная цель редукции — предотвратить переобучение, которое может произойти в случае слишком больших и сложных деревьев.

### Что такое редукция дерева?

Редукция дерева заключается в удалении некоторых ветвей дерева, которые могут быть шумными или несущественными для принятия решений. Это помогает улучшить обобщающую способность модели на новых данных, уменьшив количество ошибок на тестовом наборе данных.

### Методы редукции дерева

Существует несколько основных методов редукции дерева решений:

#### 1. Pre-pruning (предварительная редукция)

При предварительной редукции останавливаем рост дерева до того, как оно станет слишком сложным:

- Задаем максимальную глубину дерева (max_depth)
- Устанавливаем минимальное количество примеров для разбиения узла (min_samples_split)
- Определяем минимальное количество примеров в листе (min_samples_leaf)

Эти параметры помогают контролировать рост дерева на этапе обучения.

#### 2. Post-pruning (пост-редукция)

При пост-редукции сначала строим полное дерево, а затем удаляем узлы, которые не улучшают качество модели. Основные методы:

##### Reduced Error Pruning (REP)

1. Для каждого внутреннего узла:
   - Временно заменяем поддерево на лист
   - Оцениваем ошибку на валидационной выборке
   - Если ошибка не увеличилась, удаляем поддерево

Формула ошибки:
$$
Error = \frac{1}{N}\sum_{i=1}^N I(y_i \neq \hat{y_i})
$$
где $I$ - индикаторная функция, $N$ - размер валидационной выборки

##### Cost Complexity Pruning (CCP)

Минимизируем функцию стоимости:
$$
Cost_{\alpha}(T) = Error(T) + \alpha|T|
$$
где:
- $T$ - дерево
- $Error(T)$ - ошибка на обучающей выборке
- $|T|$ - количество листьев
- $\alpha$ - параметр сложности

Алгоритм:
1. Начинаем с полного дерева $T_0$
2. Для каждого $\alpha > 0$:
   - Находим поддерево $T_{\alpha}$, минимизирующее $Cost_{\alpha}(T)$
   - Добавляем пару $(\alpha, T_{\alpha})$ в последовательность
3. Выбираем оптимальное $\alpha$ по валидационной выборке

#### 3. Minimum Description Length (MDL)

Использует принцип минимальной длины описания:
$$
MDL(T) = L(T) + L(D|T)
$$
где:
- $L(T)$ - длина описания дерева в битах
- $L(D|T)$ - длина описания ошибок классификации

Для дерева $T$:
$$
L(T) = \sum_{n \in N} (log_2(|F|) + log_2(|V_f|))
$$
где:
- $N$ - множество внутренних узлов
- $|F|$ - количество признаков
- $|V_f|$ - количество возможных значений признака $f$

### Выбор метода редукции

При выборе метода редукции следует учитывать:

1. Размер данных:
   - Для больших наборов данных предпочтительнее pre-pruning
   - Для малых наборов post-pruning может дать лучшие результаты

2. Вычислительные ресурсы:
   - Pre-pruning быстрее, так как не требует построения полного дерева
   - Post-pruning требует больше вычислений, но может дать более точные результаты

3. Интерпретируемость:
   - CCP обычно дает более интерпретируемые деревья
   - MDL хорош, когда важна компактность модели

4. Баланс между точностью и обобщением:
   - REP хорошо работает, когда есть отдельная валидационная выборка
   - CCP позволяет точно контролировать компромисс между сложностью и точностью

### Пример реализации REP на Python

```python
    def prune(self, X_test: np.ndarray, y_test: np.ndarray):
        """Prune the tree using test data and Reduced Error Pruning."""

        if self.tree is None:
            raise ValueError("Tree must be fitted before pruning")

        self.tree = self._prune_recursive(self.tree, X_test, y_test)

    def _evaluate_accuracy(
        self, node: Node, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Calculate accuracy for the given node."""
        predictions = []
        for row in X:
            current = node
            while True:
                match current:
                    # Navigate until we hit a leaf
                    case LeafNode(class_=class_):
                        predictions.append(class_)
                        break

                    # Numeric feature
                    case ParentNode(
                        threshold=float() as threshold,
                        feature=feature,
                        left=left,
                        right=right,
                    ):
                        go_left = row[feature] <= threshold
                        current = left if go_left else right

                    # Categorical feature
                    case ParentNode(
                        threshold=threshold,
                        feature=feature,
                        left=left,
                        right=right,
                    ):
                        go_left = row[feature] == threshold
                        current = left if go_left else right
                    case _:
                        raise ValueError(f"Invalid node: {current}")

        return np.mean(predictions == y)

    def _should_prune(
        self, node: Node, X: np.ndarray, y: np.ndarray
    ) -> tuple[bool, float]:
        """Determine if node should be pruned by comparing accuracies."""
        accuracy_before = self._evaluate_accuracy(node, X, y)

        temp_node = LeafNode(class_=np.argmax(np.bincount(y)))

        accuracy_after = self._evaluate_accuracy(temp_node, X, y)

        return accuracy_after >= accuracy_before, accuracy_after

    def _prune_recursive(
        self, node: Node, X: np.ndarray, y: np.ndarray
    ) -> Node:
        """Recursively prune the tree."""
        match node:
            # Base case: if we're at a leaf, return
            case LeafNode():
                return node

            case ParentNode(
                threshold=threshold,
                feature=feature,
                left=left,
                right=right,
            ):
                # Recursively prune children first
                match threshold:
                    case float():
                        left_mask = X[:, feature] <= threshold
                    case _:
                        left_mask = X[:, feature] == threshold

                right_mask = ~left_mask

                if len(X[left_mask]) > 0:
                    node.left = self._prune_recursive(left, X[left_mask], y[left_mask])
                if len(X[right_mask]) > 0:
                    node.right = self._prune_recursive(right, X[right_mask], y[right_mask])

                # After pruning children, check if this node should be pruned
                should_prune, _ = self._should_prune(node, X, y)
                if should_prune:
                    return LeafNode(class_=np.argmax(np.bincount(y)))

                return node
            case _:
                raise ValueError(f"Invalid node: {node}")
```

#### Дерево до редукции

```python
def explicit_predict(feature):                                                                                     
    if feature[1] == 'male':                                                                                       
        if feature[5] <= 15.5:                                                                                     
            if feature[2] <= 32.0:                                                                                 
                if feature[2] <= 12.0:                                                                             
                    return 1                                                                                       
                else:                                                                                              
                    if feature[2] <= 26.0:                                                                         
                        return 0                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
            else:                                                                                                  
                if feature[5] <= 7.8958:                                                                           
                    if feature[2] <= 74.0:                                                                         
                        return 0                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
                else:                                                                                              
                    if feature[0] <= 2.0:                                                                          
                        return 0                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
        else:                                                                                                      
            if feature[0] <= 1.0:                                                                                  
                if feature[2] <= 42.0:                                                                             
                    if feature[2] <= 31.0:                                                                         
                        return 0                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
                else:                                                                                              
                    if feature[5] <= 35.5:                                                                         
                        return 0                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
            else:                                                                                                  
                if feature[2] <= 9.0:                                                                              
                    if feature[3] <= 1.0:                                                                          
                        return 1                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
                else:                                                                                              
                    if feature[5] <= 46.9:                                                                         
                        return 0                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
    else:                                                                                                          
        if feature[0] <= 3.0:                                                                                      
            if feature[5] <= 23.25:                                                                                
                if feature[6] == 'S':                                                                              
                    if feature[5] <= 10.5167:                                                                      
                        return 0                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
                else:                                                                                              
                    if feature[5] <= 15.2458:                                                                      
                        return 1                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
            else:                                                                                                  
                if feature[2] <= 5.0:                                                                              
                    if feature[2] <= 2.0:                                                                          
                        return 0                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
                else:                                                                                              
                    return 0                                                                                       
        else:                                                                                                      
            if feature[2] <= 27.0:                                                                                 
                if feature[2] <= 23.0:                                                                             
                    if feature[2] <= 2.0:                                                                          
                        return 0                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
                else:                                                                                              
                    if feature[2] <= 24.0:                                                                         
                        return 1                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
            else:                                                                                                  
                if feature[6] == 'C':                                                                              
                    if feature[5] <= 28.7125:                                                                      
                        return 1                                                                                   
                    else:                                                                                          
                        return 1                                                                                   
                else:                                                                                              
                    return 1  
```

#### Дерево после редукции

```python
def explicit_predict(feature):                                                                                     
    if feature[1] == 'male':                                                                                       
        if feature[5] <= 15.5:                                                                                     
            return 0                                                                                               
        else:                                                                                                      
            if feature[0] <= 1.0:                                                                                  
                if feature[2] <= 42.0:                                                                             
                    return 1                                                                                       
                else:                                                                                              
                    return 0                                                                                       
            else:                                                                                                  
                if feature[2] <= 9.0:                                                                              
                    if feature[3] <= 1.0:                                                                          
                        return 1                                                                                   
                    else:                                                                                          
                        return 0                                                                                   
                else:                                                                                              
                    return 0                                                                                       
    else:                                                                                                          
        return 1
```


Как и ожидалось, редукция дерева не изменила качество модели, но зато заметно ее упростила.

|Metric|Before Pruning|After Pruning|
|------|--------------|-------------|
|Accuracy|0.6480|0.6480|
|Precision|0.6533|0.6533|
|Recall|0.6480|0.6480|
|F1|0.6094|0.6094|

### Сравнение с sklearn

|Metric|Custom|sklearn|
|------|--------------|-------------|
|Accuracy|0.6480|0.7933|
|Precision|0.6533|0.7977|
|Recall|0.6480|0.7933|
|F1|0.6094|0.7876|

