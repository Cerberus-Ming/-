%% 1.1
% 绘制信号波形
t = -pi:0.001:pi
% syms t w
f = 0.5 * (1 + cos(t)) .* (heaviside(t + pi) - heaviside(t - pi)) % 定义信号
% F = fourier(f,t,w)
subplot(2,1,1)
plot(t,f)
grid()
xlabel('t')
ylabel('f(t)')
title('信号波形')
% 绘制信号频谱
w = -10:0.001:10
F = sin(pi.*w)./(w.*(1-w.*w)) % 定义信号的FT（手动求解得到）
subplot(2,1,2)
plot(w,F)
grid()
xlabel('w')
ylabel('F(jw)')
title('信号频谱')

%%
syms t
f = 0.5 * (1 + cos(t)) * (heaviside(t + pi) - heaviside(t - pi)); %信号f(t)
ft = fourier(f); %傅里叶变换
%绘制时域波形图
figure(1), subplot(1, 3, 1)
fplot(t, f); title('f(t) 时域波形图');
xlabel('t'); ylabel('f(t)');
axis([-5, 5, -0.1, 1.05]), grid on;
%绘制幅度谱
subplot(1, 3, 2)
fplot(abs(ft)), title('f(t) 幅度谱');
xlabel('w'); ylabel('幅度');
axis([-5, 5, -0.2, 3.3]), grid on;
%绘制相位谱
subplot(1, 3, 3)
fplot(angle(ft)), title('f(t) 相位谱');
xlabel('w'); ylabel('相位');
axis([-5, 5, -0.2, 3.3]), grid on;

%% 1.2.1
t = -pi :0.5:pi
f = 0.5 * (1 + cos(t)) % 定义函数 f(t)
% 绘制抽样信号的时域波形
subplot(3,1,1) 
stem(t, f) % 用stem函数绘制离散序列图
grid()
xlabel('t')
ylabel('f(t)')
title('0.5s抽样信号的时域波形')

w = linspace(-20, 20, 1000)
F = sin(pi.*w)./(w.*(1-w.*w)) % 定义信号的FT

% 循环计算频谱 F
for i = 1:3
    F = F + sin(pi.*(w + i * 4 * pi))./((w + i * 4 * pi).*(1-(w + i * 4 * pi).*(w + i * 4 * pi)));
    F = F + sin(pi.*(w - i * 4 * pi))./((w - i * 4 * pi).*(1-(w - i * 4 * pi).*(w - i * 4 * pi)));
end
% 绘制抽样信号的频谱
subplot(3,1,2)
plot(w, F * 2)
xlabel('w')
ylabel('F(jw)')
grid()
title('0.5s抽样信号的幅度谱')
% 绘制抽样信号的相位谱
subplot(3,1,3)
plot(w,angle(F))
xlabel('w')
ylabel('F(jw)')
grid()
title('0.5s抽样信号的相位谱')

%% 1.2.2
t = -pi :1:pi
f = 0.5 * (1 + cos(t)) % 定义函数 f(t)
% 绘制抽样信号的时域波形
subplot(3,1,1) 
stem(t, f) % 用stem函数绘制离散序列图
grid()
xlabel('t')
ylabel('f(t)')
title('1s抽样信号的时域波形')

w = linspace(-10, 10, 1000)
F = sin(pi.*w)./(w.*(1-w.*w)) % 定义信号的FT

for i=1:3
    F = F+sin(pi.*(w + i * 2 * pi))./((w + i * 2 * pi).*(1-(w + i * 2 * pi).*(w + i * 2 * pi)))
    F = F+sin(pi.*(w - i * 2 * pi))./((w - i * 2 * pi).*(1-(w - i * 2 * pi).*(w - i * 2 * pi)))
end
% 绘制抽样信号的幅度谱
subplot(3,1,2)
plot(w,F)
xlabel('w')
ylabel('F(jw)')
grid()
title('1s抽样信号的幅度谱')
% 绘制抽样信号的相位谱
subplot(3,1,3)
plot(w,angle(F))
xlabel('w')
ylabel('F(jw)')
grid()
title('1s抽样信号的相位谱')

%% 1.2.3
t = -pi :2:pi
f = 0.5 * (1 + cos(t)) % 定义函数 f(t)
% 绘制抽样信号的时域波形
subplot(3,1,1) 
stem(t, f) % 用stem函数绘制离散序列图
grid()
xlabel('t')
ylabel('f(t)')
title('2s抽样信号的时域波形')

w = linspace(-5, 5, 1000)
F = sin(pi.*w)./(w.*(1-w.*w)) % 定义信号的FT

for i=1:10
    F = F+sin(pi.*(w + i * pi))./((w + i * pi).*(1-(w + i * pi).*(w + i * pi)))
    F = F+sin(pi.*(w - i * pi))./((w - i * pi).*(1-(w - i * pi).*(w - i * pi)))
end
% 绘制抽样信号的幅度谱
subplot(3,1,2)
plot(w,F)
xlabel('w')
ylabel('F(jw)')
grid()
title('2s抽样信号的幅度谱')
% 绘制抽样信号的相位谱
subplot(3,1,3)
plot(w,angle(F))
xlabel('w')
ylabel('F(jw)')
grid()
title('2s抽样信号的相位谱')

%% 2.1.1
t = -6 : 0.01 : 6; 
f = ((1 + cos(t)) / 2) .* (heaviside(t + pi) - heaviside(t - pi)) %定义函数 f(t)

% 抽样间隔0.5s下 抽样信号通过ILPF后的信号时域波形图
Ts = 0.5; %抽样间隔
n = -6 : 6; %抽样点
nTs = n * Ts; %时间向量
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %抽样信号
w_c = 2.4; %截止频率
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %恢复信号
figure, subplot(1, 3, 1);
stem(nTs, fs)
hold on
plot(t, ft),
title('T = 0.5s')
grid on,
legend('抽样信号', '恢复信号')

% 抽样间隔1s下 抽样信号通过ILPF后的信号时域波形图
Ts = 1 %抽样间隔
nTs = -6 : Ts : 6 %时间向量
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)) %抽样信号
w_c = 2.4 %截止频率
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))) %恢复信号
subplot(1, 3, 2)
stem(nTs, fs)
hold on
plot(t, ft)
title('T = 1s')
grid on
legend('抽样信号', '恢复信号')
axis([-4, 4, -0.2, 1.2])

% 抽样间隔2s下 抽样信号通过ILPF后的信号时域波形图
Ts = 2 %抽样间隔
nTs = -pi : Ts : pi %时间向量
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)) %抽样信号
w_c = 2.4; %截止频率
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %恢复信号
subplot(1, 3, 3)
stem(nTs, fs)
hold on
plot(t, ft)
title('T = 2s')
grid on
legend('抽样信号', '恢复信号')

%% 2.1.2
t = -6 : 0.01 : 6
f = ((1 + cos(t)) / 2) .* (heaviside(t + pi) - heaviside(t - pi))  %定义函数 f(t)

% 抽样间隔0.5s下 恢复信号与原信号的绝对误差图
Ts = 0.5; %抽样间隔
n = -6 : 6; %抽样点
nTs = n * Ts; %抽样时间向量
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %抽样信号
w_c = 2.4; %截止频率
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %恢复信号
figure(1)
subplot(1, 2, 1)
plot(t, f)
hold on
plot(t, ft)
title('(T = 0.5s) 恢复信号、原始信号')
grid on
legend('原信号','恢复信号')
subplot(1, 2, 2)
plot(t, abs(f - ft))
axis([-6, 6, 0, 0.018])
title('绝对误差图')
grid on

% 抽样间隔1s下 恢复信号与原信号的绝对误差图
Ts = ; %抽样间隔
n = -6 : 6; %抽样点
nTs = n * Ts; %抽样时间向量
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %抽样信号
w_c = 2.4; %截止频率
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %恢复信号
figure(2)
subplot(1, 2, 1)
plot(t, f)
hold on
plot(t, ft)
title('(T = 1s) 恢复信号、原始信号')
grid on
legend('原信号','恢复信号')
subplot(1, 2, 2)
plot(t, abs(f - ft))
axis([-6, 6, 0, 0.018])
title('绝对误差图')
grid on

% 抽样间隔2s下 恢复信号与原信号的绝对误差图
Ts = 2; %抽样间隔
n = -pi : pi; %抽样点
nTs = n * Ts; %抽样时间向量
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %抽样信号
w_c = 2.4; %截止频率
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %恢复信号
figure(3)
subplot(1, 2, 1)
plot(t, f)
hold on
plot(t, ft)
title('(T = 2s) 恢复信号、原始信号')
grid on
legend('原信号','恢复信号')
subplot(1, 2, 2)
plot(t, abs(f - ft))
title('绝对误差图') 
grid on


%% 2.1.2

T=[0.5 1 2]; % 采样周期T
dt=0.01;
wc = 2.4;
for i=1:3
k=-pi/T(i):pi/T(i);
kT=k*T(i)
t=-pi:dt:pi;
%f(kT)
x=(1+cos(kT))/2;
%t-kT
t_kT=ones(length(k),1)*t-kT'*ones(1,length(t));
xa=x*T(i)/pi*(sin(wc*t_kT)./(t_kT));
f=(1+cos(t))/2;
%采样后的还原信号
subplot(2,3 ,i),plot(t,xa);
new_title=['采样间隔T=',num2str(T(i)),'的还原信号'];
title(new_title);
xlabel('t');
ylabel('Fs(t)');
%计算绝对误差
subplot(2,3,i+3),plot(t,abs(xa-f));
new_title=['采样间隔T=',num2str(T(i)),'的绝对误差'];
title(new_title);
xlabel('t');
ylabel('error(t)');
end

%% 2.2
T = 1; % 抽样间隔
t = -pi:T:pi;
x = 0.5*(1+cos(t)).*(heaviside(t+pi)-heaviside(t-pi));
y = filter([1 -1], [1 -0.9], x); % DAC一阶保持器恢复信号
plot(t,y,t,x);
legend('恢复信号','原信号')
xlabel('t');
ylabel('f(t)');
title('经过 DAC 一阶保持器后的恢复信号时域波形');


% 抽样间隔
T_s = 1;
% 生成时间序列
t = -pi:T_s:pi;
% 定义原始信号
f_t = 0.5 * (1 + cos(t));
% 定义保持器的参数 a 和 b
a = 1;
b = [1 0];
% 使用 filter 函数实现一阶保持器
y = filter(b, a, f_t);
% 绘制时域波形
figure;
plot(t,y,t,f_t);
legend('恢复信号','原信号')
axis([-4 4 0 1]);
grid on;