t = -4:0.01:4;
a_0 = 1/2
%a_n = 2/(pi*k)
%% 创建矩形脉冲信号
rect_pulse = 0.5 + 0.5*square(pi*(t+0.5));
%% 前 3 项之和
n = 3;
for k = 1:n
    if k == 1
        f_n = a_0;
    elseif mod(k,2) == 0
        f_n = f_n + 2/(pi*(2*k-3)) * cos((2*k-3)*pi*t);
    else
        f_n = f_n - 2/(pi*(2*k-3)) * cos((2*k-3)*pi*t);
    end
end
subplot(3,1,1);
plot(t,f_n,t,rect_pulse);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('fn(t)');
title('图1：f0+f1+f3');

%% 前 5 项之和
n = 5;
for k = 1:n
    if k == 1
        f_n = a_0;
    elseif mod(k,2) == 0
        f_n = f_n + 2/(pi*(2*k-3)) * cos((2*k-3)*pi*t);
    else
        f_n = f_n - 2/(pi*(2*k-3)) * cos((2*k-3)*pi*t);
    end
end
subplot(3,1,2);
plot(t,f_n,t,rect_pulse);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('fn(t)');
title('图2：f0+f1+f3+f5+f7');

%% 前 20 项之和
n = 20;
for k = 1:n
    if k == 1
        f_n = a_0;
    elseif mod(k,2) == 0
        f_n = f_n + 2/(pi*(2*k-3)) * cos((2*k-3)*pi*t);
    else
        f_n = f_n - 2/(pi*(2*k-3)) * cos((2*k-3)*pi*t);
    end
end
subplot(3,1,3);
plot(t,f_n,t,rect_pulse);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('fn(t)');
title('图3：f0+f1+f3+...+f35+f37');


t = -4:0.01:4;
F_0 = 1/2
%a_n = 2/(pi*k)
%% 创建矩形脉冲信号
rect_pulse = 0.5 + 0.5*square(pi*(t+0.5));
%% 前 3 项之和
n = 1;
f_n = F_0;
for k = 1:n
    if mod(k,2) == 1
        f_n = f_n + 1/(pi*(2*k-1)) * (exp(1i*(2*k-1)*pi*t)+exp(-1i*(2*k-1)*pi*t));
    else
        f_n = f_n -1/(pi*(2*k-1)) * (exp(1i*(2*k-1)*pi*t)+exp(-1i*(2*k-1)*pi*t));
    end
end
subplot(3,1,1);
plot(t,f_n,t,rect_pulse);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('fn(t)');
title('图1：f0+f1+f-1');

%% 前 7 项之和
n = 2;
f_n = F_0;
for k = 1:n
    if mod(k,2) == 1
        f_n = f_n + 1/(pi*(2*k-1)) * (exp(1i*(2*k-1)*pi*t)+exp(-1i*(2*k-1)*pi*t));
    else
        f_n = f_n -1/(pi*(2*k-1)) * (exp(1i*(2*k-1)*pi*t)+exp(-1i*(2*k-1)*pi*t));
    end
end
subplot(3,1,2);
plot(t,f_n,t,rect_pulse);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('fn(t)');
title('图2：f0+f1+f-1+f2+f-2+f3+f-3');

%% 前 20 项之和
n = 5;
f_n = F_0;
for k = 1:n
    if mod(k,2) == 1
        f_n = f_n + 1/(pi*(2*k-1)) * (exp(1i*(2*k-1)*pi*t)+exp(-1i*(2*k-1)*pi*t));
    else
        f_n = f_n -1/(pi*(2*k-1)) * (exp(1i*(2*k-1)*pi*t)+exp(-1i*(2*k-1)*pi*t));
    end
end
subplot(3,1,3);
plot(t,f_n,t,rect_pulse);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('fn(t)');
title('图3：f0+f1+f-1+f2+f-2+...+f10+f-10');

%% 符号求解傅里叶变换
syms t w
f = exp(-t).*heaviside(t);
F = fourier(f,t,w);

w=linspace(-10,10,1000);
H=1./(1j*w+1);
%% 绘制幅度图
subplot(2,1,1);
plot(w,abs(H));
xlabel('频率');
ylabel('幅度');
title('幅度谱');
%% 绘制相位图
subplot(2,1,2);
plot(w,angle(H));
xlabel('频率');
ylabel('相位');
title('相位谱')


%% 数值计算方法求单边指数函数的傅里叶变换
t = -10:0.01:10;
w = -4*pi:0.1:4*pi;
F = (exp(-t) .* heaviside(t)) * exp(-1i * t' * w ) * 0.01;%傅里叶变换值
A = abs(F);%幅度
%% 绘制幅度谱
subplot(2,1,1);
plot(w,A);
axis([-10,10,0,1])
xlabel('频率')
ylabel('幅度')
title('幅度谱');
%% 绘制相位谱
Z = angle(F);%相位
subplot(2,1,2);
plot(w,Z);
axis([-10,10,-2,2])
xlabel('频率')
ylabel('相位')
title('相位谱');

%% 对比符号求解和数值计算的频谱
W=linspace(-10,10,1000);
H=1./(1j*W+1);
%% 幅度谱对比
subplot(2,1,1);
plot(W,abs(H),w,A);
axis([-10,10,0,1])
xlabel('频率');
ylabel('幅度');
legend('符号求解','数值计算')
title('幅度谱')
%% 相位谱对比
subplot(2,1,2);
plot(W,angle(H),w,Z);
ylim([-2 2]);
axis([-10,10,0,1])
xlabel('频率');
ylabel('相位');
legend('符号求解','数值计算')
title('相位谱')

%% 符号求解傅里叶反变换
syms t w;
f = 1./(w^2+1);
F=ifourier(f,t,w);
%% 绘制波形图
t=-10:0.01:10;
f=exp(-abs(t))/2;
plot(t,f);
xlabel('t');
ylabel('y(t)');
title('波形图');


t = -10:0.01:10 %时域范围
w = -10:0.1:10; %频域范围
%% 1、门函数
G=rectpuls(t,4); %G = heaviside(t+2)-heaviside(t-2) （这个表达式求出来的幅度谱有问题）
Gf = G*exp(-1i*t'.*w)*0.01 %用数值方法求傅里叶变换
subplot(4,3,1)
plot(t,G)
xlabel('t')
ylabel('$G_(t)$', 'Interpreter', 'latex')
title('门函数 波形图');
subplot(4,3,2)
plot(w,abs(Gf))
xlabel('频率')
ylabel('幅度')
title('门函数 幅度谱');
subplot(4,3,3)
plot(w,angle(Gf))
ylim([-5 5]);
xlabel('频率')
ylabel('相位');
title('门函数 相位谱');
%% 2、三角函数
t1 = -10:0.01:10
w1 =-10:0.1:10;
S = tripuls(t,4)
Sf = S*exp(-1i*t1'*w1)*0.01 %用数值方法求傅里叶变换
subplot(4,3,4)
plot(t1,S)
xlabel('t')
ylabel('value')
title('三角函数 波形图');
subplot(4,3,5)
plot(w1,abs(Sf))
xlabel('频率')
ylabel('幅度')
title('三角函数 幅度图');
subplot(4,3,6)
plot(w1,angle(Sf))
xlabel('频率')
ylabel('相位')
title('三角函数 相位图');
%% 单边指数函数
t2 = -10:0.01:10
w2 = -10:0.1:10;
X = exp(-t2).*heaviside(t2)
Xf = X*exp(-1i*t2'*w2)*0.01 %用数值方法求傅里叶变换
subplot(4,3,7)
plot(t2,X)
xlabel('t')
ylabel('$e^{-t}\epsilon(t)$', 'Interpreter', 'latex')
title('单边指数函数 波形图')
subplot(4,3,8)
plot(w2,abs(Xf))
xlabel('频率')
ylabel('幅度')
title('单边指数函数幅度谱')
subplot(4,3,9)
plot(w2,angle(Xf))
xlabel('频率')
ylabel('相位')
title('单边指数函数相位谱')
%% 第四个函数
t3 = -10:0.01:10
w3 = -10:0.1:10;
Y = exp(-t3).*heaviside(t3)-exp(t3).*heaviside(-t3)
Yf = Y*exp(-1i*t3'*w3)*0.01 %用数值方法求傅里叶变换
subplot(4,3,10)
plot(t3,Y)
xlabel('t')
ylabel('$e^{-t}\epsilon(t)-e^{t}\epsilon(-t)$', 'Interpreter', 'latex')
title('第四个函数波形图')
subplot(4,3,11)
plot(w3,abs(Yf))
xlabel('频率')
ylabel('幅度')
title('第四个函数幅度谱')
subplot(4,3,12)
plot(w3,angle(Yf))
xlabel('频率')
ylabel('相位')
title('第四个函数相位谱')



%% 绘制G_4(t) 波形、幅度、相位图
t = -10:0.01:10 %时域范围
w = -10:0.1:10; %频域范围;
G = rectpuls(t,4); %定义G_4
Gf = G*exp(-1i*t'*w)*0.01 %用数值方法求傅里叶变换
subplot(3,3,1)
plot(t,G)
xlabel('t')
ylabel('$G_4(t)$', 'Interpreter', 'latex')
title('G_4(t) 波形图')
subplot(3,3,2)
plot(w,abs(Gf))
ylim([0 8]) % 三个函数的幅度最大是8，直观起见，幅度范围都设置为[0 8]
xlabel('频率')
ylabel('幅度')
title('G_4(t) 幅度谱')
subplot(3,3,3)
plot(w,angle(Gf))
ylim([-5 5]) % 相位范围都设置为[-5 5]
xlabel('频率')
ylabel('相位')
title('G_4(t) 相位图')
%% 绘制G_4(t/2) 波形、幅度、相位图
t1 = -10:0.01:10 %时域范围
w1 = -10:0.1:10; %频域范围
G1 = rectpuls(t1/2,4)
Gf1 = G1*exp(-1i*t1'*w1)*0.01
subplot(3,3,4)
plot(t1,G1)
xlabel('t')
ylabel('$G_4(t/2)$', 'Interpreter', 'latex')
title('G_4(t/2) 波形图')
subplot(3,3,5)
plot(w1,abs(Gf1))
ylim([0 8])
xlabel('频率')
ylabel('幅度')
title('G_4(t/2) 幅度谱')
subplot(3,3,6)
plot(w1,angle(Gf1))
ylim([-5 5])
xlabel('频率')
ylabel('相位')
title('G_4(t/2) 相位谱')
%% 绘制G_4(2t) 波形、幅度、相位图
t2=-10:0.01:10
w2 =-10:0.1:10;
G2=rectpuls(t2*2,4);
Gf2=G2*exp(-1i*t2'*w2)*0.01
subplot(3,3,7)
plot(t2,G2)
xlabel('t')
ylabel('$G_4(2t)$', 'Interpreter', 'latex')
title('G_4(2t) 波形图')
subplot(3,3,8)
plot(w2,abs(Gf2))
ylim([0 8])
xlabel('频率')
ylabel('幅度')
title('G_4(2t) 幅度谱')
subplot(3,3,9)
plot(w2,angle(Gf2))
ylim([-5 5])
xlabel('频率')
ylabel('相位')
title('G_4(2t) 相位谱')


%% 绘制G_4(t) 波形、幅度、相位图
t1= -10:0.01:10 %时域范围
w1 = -10:0.01:10;%频域范围
S =( t1+2).*(heaviside(t1+2)-heaviside(t1)).*0.5+(-t1+2).*(heaviside(t1)-heaviside(t1-2)).*0.5 %定义函数
Sf = S*exp(-1i*t1'*w1)*0.01 %数值方法计算傅里叶变换
subplot(3,3,1)
plot(t1,S)
xlabel('t')
ylabel('value')
title('三角函数 波形图')
subplot(3,3,2)
plot(w1,abs(Sf))
xlabel('频率')
ylabel('幅度')
title('三角函数幅度谱')
subplot(3,3,3)
plot(w1,angle(Sf))
ylim([-1e-10,1e-10])
xlabel('频率')
ylabel('相位')
title('三角函数 相位谱')

%% 绘制?_4(t-0.1) 波形、幅度、相位图
t2 = -10:0.01:10 %时域范围
w2 = -10:0.01:10; %频域范围
S2 = (t2+2-0.1).*(heaviside(t2-0.1+2)-heaviside(t2-0.1)).*0.5+(-t2+2+0.1).*(heaviside(t2-0.1)-heaviside(t2-2-0.1)).*0.5 %定义函数
Sf2 = S2*exp(-1i*t2'*w2)*0.01 %数值方法计算傅里叶变换
subplot(3,3,4)
plot(t2,S2)
xlabel('t')
ylabel('value')
title('三角函数右移0.1 波形图')
subplot(3,3,5)
plot(w2,abs(Sf2))
xlabel('频率')
ylabel('幅度')
title('三角函数右移0.1 幅度谱')
subplot(3,3,6)
plot(w2,angle(Sf2))
xlabel('频率')
ylabel('相位')
title('三角函数右移0.1 相位谱')

%% 绘制G_4(t-1) 波形、幅度、相位图
t3=-10:0.01:10  %时域范围
w3 =-10:0.01:10; %频域范围
S3=(t3-1+2).*(heaviside(t3-1+2)-heaviside(t3-1)).*0.5+(-t3+1+2).*(heaviside(t3-1)-heaviside(t3-1-2)).*0.5 %定义函数
Sf3=S3*exp(-1i*t3'*w3)*0.01 %数值方法计算傅里叶变换
subplot(3,3,7)
plot(t3,S3)
xlabel('t')
ylabel('value')
title('三角函数右移1 波形图')
subplot(3,3,8)
plot(w3,abs(Sf3))
xlabel('频率')
ylabel('幅度')
title('三角函数右移1 幅度谱')
subplot(3,3,9)
plot(w3,angle(Sf3))
xlabel('频率')
ylabel('相位')
title('三角函数右移1 相位谱')


%% 绘制G_4(t) 波形、幅度、相位图
t = -5:0.01:5%时域范围
w = -10:0.01:10;%频域范围
G = rectpuls(t,4); %定义函数
Gf = G*exp(-1i*t'*w)*0.01 %数值方法计算傅里叶变换
subplot(2,3,1)
plot(t,G)
xlabel('t')
ylabel('$G_4(t)$', 'Interpreter', 'latex')
title('G_4(t) 波形图')
subplot(2,3,2)
plot(w,abs(Gf))
xlabel('频率')
ylabel('幅度')
title('G_4(t) 幅度谱')
subplot(2,3,3)
plot(w,angle(Gf))
xlabel('频率')
ylabel('相位')
title('G_4(t) 相位谱')
%% 绘制G_4(t)cos(20t) 波形、幅度、相位图
t1 = -5:0.01:5 %时域范围
w1 = -10:0.01:10; %频域范围
G1 = rectpuls(t,4).*cos(20*t1) %定义函数
Gf1 = G1*exp(-1i*t1'*w1)*0.01 %数值方法计算傅里叶变换
subplot(2,3,4)
plot(t1,G1)
xlabel('t')
ylabel('$G_4(t)cos(20t)$', 'Interpreter', 'latex')
title('G_4(t)cos(20t) 波形图')
subplot(2,3,5)
plot(w1,abs(Gf1))
xlabel('频率')
ylabel('幅度')
title('G_4(t)cos(20t) 幅度谱')
subplot(2,3,6)
plot(w1,angle(Gf1))
xlabel('频率')
ylabel('相位')
title('G_4(t)cos(20t) 相位谱')
