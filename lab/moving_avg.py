import matplotlib.pyplot as plt

input_data = [15, 16, 24, 28, 18, 21, 17, 22, 25, 16, 24, 15, 23, 22]

def moving_avg(data, window_size):
    result = []
    moving_sum = sum(data[:window_size])
    result.append(moving_sum / window_size)
    for i in range(len(data) - window_size):
        moving_sum = moving_sum + (data[i + window_size] - data[i])
        result.append(moving_sum / window_size)
    return result

window_size = 3

ma = moving_avg(input_data, window_size)
print(ma)

x_ma = list(range(window_size - 1, len(input_data)))

plt.figure(figsize=(10, 5))
plt.plot(input_data, label='Original Data', marker='o')
plt.plot(x_ma, ma, label=f'{window_size}-Point Moving Average', color='red', marker='o', linestyle='--')
plt.title('Moving Average Plot')
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
