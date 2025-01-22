%% 1.1
%定义系统函数
a = [1 1 1]
b = [1]
w = logspace(-2, 2, 1000)
h = freqs(b,a,w) 
%绘制幅频特性曲线
subplot(2,1,1)
plot(w,abs(h))
grid
xlim([0 10])
xlabel('w')
ylabel('|H(jw)|')
title('幅频特性曲线')
%绘制相频特性曲线
subplot(2,1,2);
plot(w,angle(h))
grid
xlabel('w')
xlim([0 30])
ylabel('相位')
title('相频特性曲线')

%% 1.2
t = 0:0.01:10;  % 定义时间范围
fs = 1 / (t(2) - t(1));  % 定义采样频率
f = cos(t) + cos(10 * t) %定义输入信号
t_impulse = 0:0.01:5;  % 冲激响应的时间范围
h_impulse = impulse([1], [1, 1, 1], t_impulse);  % 计算系统的冲激响应
y = conv(f, h_impulse, 'same');
figure;
% 绘制输入信号
subplot(2,1,1);
plot(t, f);
xlabel('时间 (s)');
ylabel('幅度');
title('输入信号');
% 绘制响应信号
subplot(2,1,2);
plot(t, y);
xlabel('时间 (s)');
ylabel('幅度');
title('响应信号');

%% 2.1.1
syms t
f1 = exp(-2*t)*heaviside(t) %定义函数
F1 = laplace(f) %使用函数计算拉普拉斯变换

%% 2.1.2
syms t
f2 = dirac(t)+exp(2*t)-4/3*exp(-t)*heaviside(t) %定义函数
F2 = laplace(f) %使用函数计算拉普拉斯变换

%% 2.2.1
syms s 
F1 = (4*s + 5) / (s^2 + 5*s + 6); %拉普拉斯变换
f1 = ilaplace(F1) %使用函数计算拉普拉斯反变换

%% 2.2.2
syms s
F2 = 3*s/((s+4)*(s+2)) %拉普拉斯变换
f2 = ilaplace(F2) %使用函数计算拉普拉斯反变换

%% 2.3
%通过for循环分别绘制极点处于-1.5，-0.5，0，0.5，1.5时的极零图及对应的冲激响应函数
T=[1.5 0.5 0 -0.5 -1.5]
for i=1:5
    b=[1]
    a=[1 T(i)]
    t=0:0.01:2*pi
    % 绘制极零图
    sys=tf(b,a)
    subplot(2,5,i)
    pzmap(sys)
    % 绘制冲激响应
    y = impulse(b,a,t)
    subplot(2,5,i+5)
    plot(t,y)
    xlabel('t')
    ylabel('h(t)')
    title('冲激响应')
end

%% 3.1
syms k z
f = 2^(k-1).*stepfun(k,0)  %定义函数
F = ztrans(f,k,z) %使用函数计算Z变换

%% 3.2
syms z
Fz = (2*z*z-0.5*z)/(z*z-0.5*z-0.5) %定义有始信号的Z变换
f = iztrans(Fz) %使用函数计算反变换

%% 3.3.1
% 定义的参数 a 和 b
b = [1]
a = [1 -0.8]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_1(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_1(k)')
title('时域单位函数响应波形图')

%% 3.3.2
% 定义的参数 a 和 b
b = [1]
a = [1 -1]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_2(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_2(k)')
title('时域单位函数响应波形图')

%% 3.3.3
% 定义的参数 a 和 b
b = [1]
a = [1 -1.2]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_3(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_3(k)')
title('时域单位函数响应波形图')

%% 3.3.4
% 定义的参数 a 和 b
b = [1]
a = [1 0.8]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_4(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_4(k)')
title('时域单位函数响应波形图')

%% 3.3.5
% 定义的参数 a 和 b
b = [1]
a = [1 -1.2 0.72]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_5(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_5(k)')
title('时域单位函数响应波形图')

%% 3.3.6
% 定义的参数 a 和 b
b = [1]
a = [1 -1.6 1]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_6(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_6(k)')
title('时域单位函数响应波形图')

%% 3.3.7
% 定义的参数 a 和 b
b = [1]
a = [1 -2 1.36]
%绘制极零点图
subplot(2,1,1)
zplane(b,a)
legend('零点','极点')
title('H_7(z)零极点图')
%绘制时域单位函数响应波形图
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)的波形
stem(k,h)
xlabel('k')
ylabel('h_7(k)')
title('时域单位函数响应波形图')