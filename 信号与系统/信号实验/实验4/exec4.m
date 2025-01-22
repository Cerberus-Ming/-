%% 1.1
%����ϵͳ����
a = [1 1 1]
b = [1]
w = logspace(-2, 2, 1000)
h = freqs(b,a,w) 
%���Ʒ�Ƶ��������
subplot(2,1,1)
plot(w,abs(h))
grid
xlim([0 10])
xlabel('w')
ylabel('|H(jw)|')
title('��Ƶ��������')
%������Ƶ��������
subplot(2,1,2);
plot(w,angle(h))
grid
xlabel('w')
xlim([0 30])
ylabel('��λ')
title('��Ƶ��������')

%% 1.2
t = 0:0.01:10;  % ����ʱ�䷶Χ
fs = 1 / (t(2) - t(1));  % �������Ƶ��
f = cos(t) + cos(10 * t) %���������ź�
t_impulse = 0:0.01:5;  % �弤��Ӧ��ʱ�䷶Χ
h_impulse = impulse([1], [1, 1, 1], t_impulse);  % ����ϵͳ�ĳ弤��Ӧ
y = conv(f, h_impulse, 'same');
figure;
% ���������ź�
subplot(2,1,1);
plot(t, f);
xlabel('ʱ�� (s)');
ylabel('����');
title('�����ź�');
% ������Ӧ�ź�
subplot(2,1,2);
plot(t, y);
xlabel('ʱ�� (s)');
ylabel('����');
title('��Ӧ�ź�');

%% 2.1.1
syms t
f1 = exp(-2*t)*heaviside(t) %���庯��
F1 = laplace(f) %ʹ�ú�������������˹�任

%% 2.1.2
syms t
f2 = dirac(t)+exp(2*t)-4/3*exp(-t)*heaviside(t) %���庯��
F2 = laplace(f) %ʹ�ú�������������˹�任

%% 2.2.1
syms s 
F1 = (4*s + 5) / (s^2 + 5*s + 6); %������˹�任
f1 = ilaplace(F1) %ʹ�ú�������������˹���任

%% 2.2.2
syms s
F2 = 3*s/((s+4)*(s+2)) %������˹�任
f2 = ilaplace(F2) %ʹ�ú�������������˹���任

%% 2.3
%ͨ��forѭ���ֱ���Ƽ��㴦��-1.5��-0.5��0��0.5��1.5ʱ�ļ���ͼ����Ӧ�ĳ弤��Ӧ����
T=[1.5 0.5 0 -0.5 -1.5]
for i=1:5
    b=[1]
    a=[1 T(i)]
    t=0:0.01:2*pi
    % ���Ƽ���ͼ
    sys=tf(b,a)
    subplot(2,5,i)
    pzmap(sys)
    % ���Ƴ弤��Ӧ
    y = impulse(b,a,t)
    subplot(2,5,i+5)
    plot(t,y)
    xlabel('t')
    ylabel('h(t)')
    title('�弤��Ӧ')
end

%% 3.1
syms k z
f = 2^(k-1).*stepfun(k,0)  %���庯��
F = ztrans(f,k,z) %ʹ�ú�������Z�任

%% 3.2
syms z
Fz = (2*z*z-0.5*z)/(z*z-0.5*z-0.5) %������ʼ�źŵ�Z�任
f = iztrans(Fz) %ʹ�ú������㷴�任

%% 3.3.1
% ����Ĳ��� a �� b
b = [1]
a = [1 -0.8]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_1(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_1(k)')
title('ʱ��λ������Ӧ����ͼ')

%% 3.3.2
% ����Ĳ��� a �� b
b = [1]
a = [1 -1]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_2(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_2(k)')
title('ʱ��λ������Ӧ����ͼ')

%% 3.3.3
% ����Ĳ��� a �� b
b = [1]
a = [1 -1.2]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_3(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_3(k)')
title('ʱ��λ������Ӧ����ͼ')

%% 3.3.4
% ����Ĳ��� a �� b
b = [1]
a = [1 0.8]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_4(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_4(k)')
title('ʱ��λ������Ӧ����ͼ')

%% 3.3.5
% ����Ĳ��� a �� b
b = [1]
a = [1 -1.2 0.72]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_5(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_5(k)')
title('ʱ��λ������Ӧ����ͼ')

%% 3.3.6
% ����Ĳ��� a �� b
b = [1]
a = [1 -1.6 1]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_6(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_6(k)')
title('ʱ��λ������Ӧ����ͼ')

%% 3.3.7
% ����Ĳ��� a �� b
b = [1]
a = [1 -2 1.36]
%���Ƽ����ͼ
subplot(2,1,1)
zplane(b,a)
legend('���','����')
title('H_7(z)�㼫��ͼ')
%����ʱ��λ������Ӧ����ͼ
subplot(2,1,2)
[h,k] = impz(b,a) %h(k)�Ĳ���
stem(k,h)
xlabel('k')
ylabel('h_7(k)')
title('ʱ��λ������Ӧ����ͼ')