Для выполнения первой лабораторной работы были выбраны датасеты с видами рыб и видами цветов ириса. Целевой признак, в котором и хранится разделение по кластерам, был удален. Количество кластеров было подобрано при помощи дендрограммы. Для датасета с видами рыб оно составило 9 кластеров, для датасета с видами ириса оно составило 4 кластера.

Бло реализовано три алгоритма кластеризации: иерархический, em и dbcscan. 

1. Точность иерархического алгоритма, реализованного вручную, совпадает с точностью библиотечного алгоритма. Также были построены дендрограммы по результатам ручной кластеризации. Данные дендрограммы полностью совпадают с библиотечными дендрограммамми. Однако время работы сильно отличается: составляет намного больше, чем время работы библиотечного алгоритма.
![image](https://github.com/user-attachments/assets/6703c771-df20-408b-aea0-e0b3b27364bc)

Дендрограмма ручного алгоритма (датасет с рыбами)

![image](https://github.com/user-attachments/assets/a6806f1e-5db9-48e1-9830-d66a7c436102)

Дендрограмма автоматического алгоритма (датасет с рыбами)

![image](https://github.com/user-attachments/assets/4c7f2fcc-9021-4589-b5eb-44ecdc93a526)
![image](https://github.com/user-attachments/assets/a2b0395a-09c0-4d1a-88f9-1898b42bcb95)
![image](https://github.com/user-attachments/assets/cf3aba97-dcd7-42b8-8e25-537503a6b23e)

Сравнение метрик на датасете с рыбами

![image](https://github.com/user-attachments/assets/487e2027-0ee4-45a6-9ed3-18ab0fa93054)
![image](https://github.com/user-attachments/assets/00782529-4e38-4b48-8b38-a312b294d9d4)

Сравнение разделения на датасете с рыбами (график выше ручная кластеризация, график ниже библиотечная)

![image](https://github.com/user-attachments/assets/5e139d90-3972-4fd4-ae39-06e54e94c701)

Дендрограмма ручного алгоритма (датасет с ирисами)

![image](https://github.com/user-attachments/assets/fbdf2ab2-f9ba-43df-a355-2f6e4eebbb15)

Дендрограмма автоматического алгоритма (датасет с ирисами)

![image](https://github.com/user-attachments/assets/361816de-8f76-4c14-8f22-acaae6c12ee0)
![image](https://github.com/user-attachments/assets/b36cbe65-f76f-4a9b-b8d9-b210170e3ee5)
![image](https://github.com/user-attachments/assets/5d77f92e-e2c8-4040-9f9e-3cca62bd70ea)

Сравнение метрик на датасете с ирисами

![image](https://github.com/user-attachments/assets/d73fa578-5d98-4a52-8fc1-72def80f700a)
![image](https://github.com/user-attachments/assets/e7f27183-52c4-4efc-b77d-a204941ce796)

Сравнение разделения на датасете с ирисами (график выше ручная кластеризация, график ниже библиотечная)

2. Точность em алгоритма несколько отличается от библиотечного алгоритма: это связано с тем, что алгоритм использует рандомные начальные приближения кластеров и является вероятностным, за счет каждый раз он срабатывает по разному. Однако время работы библиотечного алгоритма и алгоритма, написанного вручную, сравнимо.
   
![image](https://github.com/user-attachments/assets/b39411d1-1086-4947-b3ec-a34a0c75b799)
![image](https://github.com/user-attachments/assets/9c0677fc-8f0b-432b-83f7-7e7fa119ca84)
![image](https://github.com/user-attachments/assets/9e8adc20-3040-4bda-b301-1b79c200a18e)

Сравнение метрик на датасете с рыбами

![image](https://github.com/user-attachments/assets/6ee15756-173f-4b11-8cda-6efa0d033c6c)
![image](https://github.com/user-attachments/assets/0370a1bc-c841-462c-aad0-4f11d09c45ae)

Сравнение разделения на датасете с рыбами (график выше ручная кластеризация, график ниже библиотечная)

![image](https://github.com/user-attachments/assets/191a7fe7-c919-4396-8540-49949011f32f)
![image](https://github.com/user-attachments/assets/2fdc49e9-c561-406e-b878-01602d070c4b)
![image](https://github.com/user-attachments/assets/f2f764f0-7f54-4f88-a54d-a4cad93e0d3c)

Сравнение метрик на датасете с ирисами

![image](https://github.com/user-attachments/assets/0a811b8e-e721-4bb5-aa49-c57cd8a54158)
![image](https://github.com/user-attachments/assets/919e1915-7a62-4bf9-9ae9-c4b11baad28d)

Сравнение разделения на датасете с ирисами (график выше ручная кластеризация, график ниже библиотечная)

3. Точность dbscan алгоритма практически полностью совпадает с точностью библиотечного алгоритма. Однако время работы библиотечного алгоритма также составляет намного меньше.

![image](https://github.com/user-attachments/assets/6041dd0d-bcf5-4be2-88d3-727dad6611da)
![image](https://github.com/user-attachments/assets/175e59be-9e2e-49cb-95bf-9f5fd2cbbcea)
![image](https://github.com/user-attachments/assets/70a1592f-7831-407d-b761-045bf95081cd)

Сравнение метрик на датасете с рыбами

![image](https://github.com/user-attachments/assets/7527a889-0713-447b-b7e3-7c78e74b79f9)
![image](https://github.com/user-attachments/assets/753676f7-6f01-4e1c-99c8-020f957b64dc)
Сравнение разделения на датасете с рыбами (график выше ручная кластеризация, график ниже библиотечная)

![image](https://github.com/user-attachments/assets/1e4436b7-c894-4fd6-b991-1570d6b79940)
![image](https://github.com/user-attachments/assets/b917f9e4-296c-4a77-b303-31a99d9ee7ac)
![image](https://github.com/user-attachments/assets/978a53df-70ac-4450-aebd-2d5a3ce9f543)

Сравнение метрик на датасете с ирисами

![image](https://github.com/user-attachments/assets/7db774d3-2f3f-4420-bbf3-f4294d696f0d)
![image](https://github.com/user-attachments/assets/a865af70-3495-4523-92ca-ab8c42c93378)

Сравнение разделения на датасете с ирисами (график выше ручная кластеризация, график ниже библиотечная)

Итого, наиболее предпочтительным алгоритмом из всех написанных является иерархический, так как точность данного алгоритма сопоставима с библиотечным, а также метрики кластеризации являются оптимальными.
