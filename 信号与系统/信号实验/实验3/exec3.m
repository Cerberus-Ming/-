%% 1.1
% �����źŲ���
t = -pi:0.001:pi
% syms t w
f = 0.5 * (1 + cos(t)) .* (heaviside(t + pi) - heaviside(t - pi)) % �����ź�
% F = fourier(f,t,w)
subplot(2,1,1)
plot(t,f)
grid()
xlabel('t')
ylabel('f(t)')
title('�źŲ���')
% �����ź�Ƶ��
w = -10:0.001:10
F = sin(pi.*w)./(w.*(1-w.*w)) % �����źŵ�FT���ֶ����õ���
subplot(2,1,2)
plot(w,F)
grid()
xlabel('w')
ylabel('F(jw)')
title('�ź�Ƶ��')

%%
syms t
f = 0.5 * (1 + cos(t)) * (heaviside(t + pi) - heaviside(t - pi)); %�ź�f(t)
ft = fourier(f); %����Ҷ�任
%����ʱ����ͼ
figure(1), subplot(1, 3, 1)
fplot(t, f); title('f(t) ʱ����ͼ');
xlabel('t'); ylabel('f(t)');
axis([-5, 5, -0.1, 1.05]), grid on;
%���Ʒ�����
subplot(1, 3, 2)
fplot(abs(ft)), title('f(t) ������');
xlabel('w'); ylabel('����');
axis([-5, 5, -0.2, 3.3]), grid on;
%������λ��
subplot(1, 3, 3)
fplot(angle(ft)), title('f(t) ��λ��');
xlabel('w'); ylabel('��λ');
axis([-5, 5, -0.2, 3.3]), grid on;

%% 1.2.1
t = -pi :0.5:pi
f = 0.5 * (1 + cos(t)) % ���庯�� f(t)
% ���Ƴ����źŵ�ʱ����
subplot(3,1,1) 
stem(t, f) % ��stem����������ɢ����ͼ
grid()
xlabel('t')
ylabel('f(t)')
title('0.5s�����źŵ�ʱ����')

w = linspace(-20, 20, 1000)
F = sin(pi.*w)./(w.*(1-w.*w)) % �����źŵ�FT

% ѭ������Ƶ�� F
for i = 1:3
    F = F + sin(pi.*(w + i * 4 * pi))./((w + i * 4 * pi).*(1-(w + i * 4 * pi).*(w + i * 4 * pi)));
    F = F + sin(pi.*(w - i * 4 * pi))./((w - i * 4 * pi).*(1-(w - i * 4 * pi).*(w - i * 4 * pi)));
end
% ���Ƴ����źŵ�Ƶ��
subplot(3,1,2)
plot(w, F * 2)
xlabel('w')
ylabel('F(jw)')
grid()
title('0.5s�����źŵķ�����')
% ���Ƴ����źŵ���λ��
subplot(3,1,3)
plot(w,angle(F))
xlabel('w')
ylabel('F(jw)')
grid()
title('0.5s�����źŵ���λ��')

%% 1.2.2
t = -pi :1:pi
f = 0.5 * (1 + cos(t)) % ���庯�� f(t)
% ���Ƴ����źŵ�ʱ����
subplot(3,1,1) 
stem(t, f) % ��stem����������ɢ����ͼ
grid()
xlabel('t')
ylabel('f(t)')
title('1s�����źŵ�ʱ����')

w = linspace(-10, 10, 1000)
F = sin(pi.*w)./(w.*(1-w.*w)) % �����źŵ�FT

for i=1:3
    F = F+sin(pi.*(w + i * 2 * pi))./((w + i * 2 * pi).*(1-(w + i * 2 * pi).*(w + i * 2 * pi)))
    F = F+sin(pi.*(w - i * 2 * pi))./((w - i * 2 * pi).*(1-(w - i * 2 * pi).*(w - i * 2 * pi)))
end
% ���Ƴ����źŵķ�����
subplot(3,1,2)
plot(w,F)
xlabel('w')
ylabel('F(jw)')
grid()
title('1s�����źŵķ�����')
% ���Ƴ����źŵ���λ��
subplot(3,1,3)
plot(w,angle(F))
xlabel('w')
ylabel('F(jw)')
grid()
title('1s�����źŵ���λ��')

%% 1.2.3
t = -pi :2:pi
f = 0.5 * (1 + cos(t)) % ���庯�� f(t)
% ���Ƴ����źŵ�ʱ����
subplot(3,1,1) 
stem(t, f) % ��stem����������ɢ����ͼ
grid()
xlabel('t')
ylabel('f(t)')
title('2s�����źŵ�ʱ����')

w = linspace(-5, 5, 1000)
F = sin(pi.*w)./(w.*(1-w.*w)) % �����źŵ�FT

for i=1:10
    F = F+sin(pi.*(w + i * pi))./((w + i * pi).*(1-(w + i * pi).*(w + i * pi)))
    F = F+sin(pi.*(w - i * pi))./((w - i * pi).*(1-(w - i * pi).*(w - i * pi)))
end
% ���Ƴ����źŵķ�����
subplot(3,1,2)
plot(w,F)
xlabel('w')
ylabel('F(jw)')
grid()
title('2s�����źŵķ�����')
% ���Ƴ����źŵ���λ��
subplot(3,1,3)
plot(w,angle(F))
xlabel('w')
ylabel('F(jw)')
grid()
title('2s�����źŵ���λ��')

%% 2.1.1
t = -6 : 0.01 : 6; 
f = ((1 + cos(t)) / 2) .* (heaviside(t + pi) - heaviside(t - pi)) %���庯�� f(t)

% �������0.5s�� �����ź�ͨ��ILPF����ź�ʱ����ͼ
Ts = 0.5; %�������
n = -6 : 6; %������
nTs = n * Ts; %ʱ������
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %�����ź�
w_c = 2.4; %��ֹƵ��
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %�ָ��ź�
figure, subplot(1, 3, 1);
stem(nTs, fs)
hold on
plot(t, ft),
title('T = 0.5s')
grid on,
legend('�����ź�', '�ָ��ź�')

% �������1s�� �����ź�ͨ��ILPF����ź�ʱ����ͼ
Ts = 1 %�������
nTs = -6 : Ts : 6 %ʱ������
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)) %�����ź�
w_c = 2.4 %��ֹƵ��
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))) %�ָ��ź�
subplot(1, 3, 2)
stem(nTs, fs)
hold on
plot(t, ft)
title('T = 1s')
grid on
legend('�����ź�', '�ָ��ź�')
axis([-4, 4, -0.2, 1.2])

% �������2s�� �����ź�ͨ��ILPF����ź�ʱ����ͼ
Ts = 2 %�������
nTs = -pi : Ts : pi %ʱ������
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)) %�����ź�
w_c = 2.4; %��ֹƵ��
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %�ָ��ź�
subplot(1, 3, 3)
stem(nTs, fs)
hold on
plot(t, ft)
title('T = 2s')
grid on
legend('�����ź�', '�ָ��ź�')

%% 2.1.2
t = -6 : 0.01 : 6
f = ((1 + cos(t)) / 2) .* (heaviside(t + pi) - heaviside(t - pi))  %���庯�� f(t)

% �������0.5s�� �ָ��ź���ԭ�źŵľ������ͼ
Ts = 0.5; %�������
n = -6 : 6; %������
nTs = n * Ts; %����ʱ������
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %�����ź�
w_c = 2.4; %��ֹƵ��
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %�ָ��ź�
figure(1)
subplot(1, 2, 1)
plot(t, f)
hold on
plot(t, ft)
title('(T = 0.5s) �ָ��źš�ԭʼ�ź�')
grid on
legend('ԭ�ź�','�ָ��ź�')
subplot(1, 2, 2)
plot(t, abs(f - ft))
axis([-6, 6, 0, 0.018])
title('�������ͼ')
grid on

% �������1s�� �ָ��ź���ԭ�źŵľ������ͼ
Ts = ; %�������
n = -6 : 6; %������
nTs = n * Ts; %����ʱ������
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %�����ź�
w_c = 2.4; %��ֹƵ��
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %�ָ��ź�
figure(2)
subplot(1, 2, 1)
plot(t, f)
hold on
plot(t, ft)
title('(T = 1s) �ָ��źš�ԭʼ�ź�')
grid on
legend('ԭ�ź�','�ָ��ź�')
subplot(1, 2, 2)
plot(t, abs(f - ft))
axis([-6, 6, 0, 0.018])
title('�������ͼ')
grid on

% �������2s�� �ָ��ź���ԭ�źŵľ������ͼ
Ts = 2; %�������
n = -pi : pi; %������
nTs = n * Ts; %����ʱ������
fs = ((1 + cos(nTs)) / 2) .* (heaviside(nTs + pi) - heaviside(nTs - pi)); %�����ź�
w_c = 2.4; %��ֹƵ��
ft = (Ts * w_c / pi) * fs * sinc((w_c ./ pi) * (ones(length(nTs), 1) * t - nTs' * ones(1,length(t)))); %�ָ��ź�
figure(3)
subplot(1, 2, 1)
plot(t, f)
hold on
plot(t, ft)
title('(T = 2s) �ָ��źš�ԭʼ�ź�')
grid on
legend('ԭ�ź�','�ָ��ź�')
subplot(1, 2, 2)
plot(t, abs(f - ft))
title('�������ͼ') 
grid on


%% 2.1.2

T=[0.5 1 2]; % ��������T
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
%������Ļ�ԭ�ź�
subplot(2,3 ,i),plot(t,xa);
new_title=['�������T=',num2str(T(i)),'�Ļ�ԭ�ź�'];
title(new_title);
xlabel('t');
ylabel('Fs(t)');
%����������
subplot(2,3,i+3),plot(t,abs(xa-f));
new_title=['�������T=',num2str(T(i)),'�ľ������'];
title(new_title);
xlabel('t');
ylabel('error(t)');
end

%% 2.2
T = 1; % �������
t = -pi:T:pi;
x = 0.5*(1+cos(t)).*(heaviside(t+pi)-heaviside(t-pi));
y = filter([1 -1], [1 -0.9], x); % DACһ�ױ������ָ��ź�
plot(t,y,t,x);
legend('�ָ��ź�','ԭ�ź�')
xlabel('t');
ylabel('f(t)');
title('���� DAC һ�ױ�������Ļָ��ź�ʱ����');


% �������
T_s = 1;
% ����ʱ������
t = -pi:T_s:pi;
% ����ԭʼ�ź�
f_t = 0.5 * (1 + cos(t));
% ���屣�����Ĳ��� a �� b
a = 1;
b = [1 0];
% ʹ�� filter ����ʵ��һ�ױ�����
y = filter(b, a, f_t);
% ����ʱ����
figure;
plot(t,y,t,f_t);
legend('�ָ��ź�','ԭ�ź�')
axis([-4 4 0 1]);
grid on;