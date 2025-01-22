%% t的取值范围为0-2，每隔0.01有一个取值点
t = 0:0.01:2;
%% 函数公式，heaviside为阶跃函数，注意点乘
x = sin(2*pi*t) .* (heaviside(t) - heaviside(t-4));
%% 画图
plot(t,x);
%% 设置y轴范围，便于显示
ylim([-1.1 1.1]);
%% 设置x y轴，图像名称
xlabel('t');
ylabel('x(t)');
title('$x(t) = \sin(2\pi t)[u(t) - u(t-4)]$', 'Interpreter', 'latex');

%% t的取值范围为0-2，每隔0.01有一个取值点
t =0:0.01:2;
%% 函数公式，heaviside为阶跃函数，exp为指数函数，注意点乘
h = exp(-t) .* heaviside(t) - exp(-2*t) .* heaviside(t);
%% 画图
plot(t,h);
%% 设置y轴范围，便于显示
ylim([0 0.28]);
%% 设置x y轴，图像名称
xlabel('t');
ylabel('h(t)');
title('$h(t) = \exp(-t)u(t) - \exp(-2t)u(t)$', 'Interpreter', 'latex');

%% t的取值范围为-4-+4，每隔0.01有一个取值点
t = -4:0.01:4;
%% 用阶跃函数实现门函数
y = 2*heaviside(t+2) - 2*heaviside(t-2);
%% 画图
plot(t,y);
%% 设置y轴范围，便于显示
ylim([0 2.1]);
%% 设置x y轴，图像名称
xlabel('t');
ylabel('y(t)');
title('$y(t) = 2u(t+2) - 2u(t-2)$', 'Interpreter', 'latex');


%% t的取值范围为-2-+2，每隔0.01有一个取值点
t = -2:0.01:2;
%% 定义函数
G1 = heaviside(t+0.5) - heaviside(t-0.5);%G1(t)
y1 = heaviside(2*t+0.5) - heaviside(2*t-0.5);%y(2t)
y2 = heaviside(t/2+0.5) - heaviside(t/2-0.5);%y(t/2)
y3 = heaviside((2-2*t)+0.5) - heaviside((2-2*t)-0.5);%y(2-2t)
%% 绘制y(t),y(2t)
%% subplot(m, n, p) 将图形窗口分成 m 行 n 列的子图网格,当前绘图为第 p 个子图
subplot(3,1,1);
plot(t,G1,t,y1);
ylim([0 1.1]);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('y(t)');
legend('y(t)','y(2t)');
title('图1 y(t),y(2t)');
%% 绘制y(t),y(t/2)
subplot(3,1,2);
plot(t,G1,t,y2);
ylim([0 1.1]);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('y(t)');
legend('y(t)','y(t/2)');
title('图2 y(t),y(t/2)');
%% 绘制y(t),y(2-2t)
subplot(3,1,3);
plot(t,G1,t,y3);
ylim([0 1.1]);
%% 设置xy坐标轴，子图名称
xlabel('t');
ylabel('y(t)');
legend('y(t),y(2-2t)');
title('图3 y(t),y(2-2t)');


%% t的取值范围为-100-+100，每隔0.01有一个取值点
t = -100:0.01:100;
%% 定义函数
y = cos(t) + cos(pi*t/4);
%% 绘制函数
plot(t,y);
%% 设置坐标轴，图像名称
xlabel('t');
ylabel('y(t)');
title( '$y(t) = \cos(t) + \cos(\pi t/4 )$', 'Interpreter', 'latex');

%% t的取值范围为-10-+10，每隔0.01有一个取值点
t = -10:0.01:10;
%% 定义函数
y = sin(pi*t) + cos(2*pi*t);
%% 绘制函数
plot(t,y);
%% 设置坐标轴，图像名称
xlabel('t');
ylabel('y(t)');
title( '$y(t) = \sin(t) + \cos(2\pi t )$', 'Interpreter', 'latex');


t = 0:0.01:20;
y1 = exp(-2 * t) .* heaviside(t);
y2 = exp(-t) .* heaviside(t);
%% 使用conv将y1和y2进行卷积
y = conv(y1, y2) .* 0.01;%由于计算是离散的点，卷积后需要乘以步长
k = 2*length(t)-1;
k1 = linspace(2*t(1),2*t(end),k);
%% 绘制conv函数获得的图像
subplot(3,1,1);
plot(k1, y);
xlabel('t');
ylabel('y(t)');
title('图1 数值计算结果')
legend('数值');
%% 绘制理论值图像
t_theory = 0:0.01:40;
y_theory = (exp(-t_theory) - exp(-2*t_theory));
subplot(3,1,2);
plot(t_theory, y_theory);
xlabel('t');
ylabel('y(t)');
title('图2 理论推导结果');
legend('理论');
%% 把理论值与问题2中的数值计算结果画到一张图中
subplot(3,1,3);
plot(k1,y,t_theory,y_theory);
xlabel('t');
ylabel('y(t)');
title('图3 数值计算和理论推导对比');
legend('数值','理论');%用legend语句加图例


t=0:0.01:20
%% 连续时间LTI系统H，它通过tf(b, a)函数
%% b、a分别为微分方程右端和左端各项的系数向量.
H=tf([1],[1 3 2]);
e=exp(-2*t).*heaviside(t);%定义输入信号
y=lsim(H,e,t);%使用lsim函数求出零状态响应
%% 绘制数值计算图像
subplot(3,1,1);
plot(t,y);
xlabel('t');
ylabel('r_{zs}');
title('图1 数值计算结果');
legend('数值');
%% 绘制理论推导图像
y1 = exp(-t) - (1+t).*exp(-2.*t);
subplot(3,1,2);
plot(t,y1);
xlabel('t');
ylabel('r_{zs}');
title('图2 理论推导结果');
legend('理论');
%% 理论和仿真对比
subplot(3,1,3);
plot(t,y,t,y1);
xlabel('t');
ylabel('r_{zs}');
title('图3 数值计算和理论推导对比');
legend('数值','理论')