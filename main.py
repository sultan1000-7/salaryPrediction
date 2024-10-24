a = int(input("Введите число: "))
b = int(input("Введите число: "))

op = input("Введите операцию: ")

if op == "+":
    print(f'Ответ: {a + b}')
elif op == "-":
    print(f'Ответ: {a - b}')
elif op == "/":
    print(f'Ответ: {a / b}')
elif op == "*":
    print(f"Ответ: {a * b}")
else:
    print("Неверная операция")