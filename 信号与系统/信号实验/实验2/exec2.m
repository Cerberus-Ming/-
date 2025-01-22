t = -4:0.01:4;
a_0 = 1/2
%a_n = 2/(pi*k)
%% �������������ź�
rect_pulse = 0.5 + 0.5*square(pi*(t+0.5));
%% ǰ 3 ��֮��
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
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('fn(t)');
title('ͼ1��f0+f1+f3');

%% ǰ 5 ��֮��
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
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('fn(t)');
title('ͼ2��f0+f1+f3+f5+f7');

%% ǰ 20 ��֮��
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
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('fn(t)');
title('ͼ3��f0+f1+f3+...+f35+f37');


t = -4:0.01:4;
F_0 = 1/2
%a_n = 2/(pi*k)
%% �������������ź�
rect_pulse = 0.5 + 0.5*square(pi*(t+0.5));
%% ǰ 3 ��֮��
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
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('fn(t)');
title('ͼ1��f0+f1+f-1');

%% ǰ 7 ��֮��
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
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('fn(t)');
title('ͼ2��f0+f1+f-1+f2+f-2+f3+f-3');

%% ǰ 20 ��֮��
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
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('fn(t)');
title('ͼ3��f0+f1+f-1+f2+f-2+...+f10+f-10');

%% ������⸵��Ҷ�任
syms t w
f = exp(-t).*heaviside(t);
F = fourier(f,t,w);

w=linspace(-10,10,1000);
H=1./(1j*w+1);
%% ���Ʒ���ͼ
subplot(2,1,1);
plot(w,abs(H));
xlabel('Ƶ��');
ylabel('����');
title('������');
%% ������λͼ
subplot(2,1,2);
plot(w,angle(H));
xlabel('Ƶ��');
ylabel('��λ');
title('��λ��')


%% ��ֵ���㷽���󵥱�ָ�������ĸ���Ҷ�任
t = -10:0.01:10;
w = -4*pi:0.1:4*pi;
F = (exp(-t) .* heaviside(t)) * exp(-1i * t' * w ) * 0.01;%����Ҷ�任ֵ
A = abs(F);%����
%% ���Ʒ�����
subplot(2,1,1);
plot(w,A);
axis([-10,10,0,1])
xlabel('Ƶ��')
ylabel('����')
title('������');
%% ������λ��
Z = angle(F);%��λ
subplot(2,1,2);
plot(w,Z);
axis([-10,10,-2,2])
xlabel('Ƶ��')
ylabel('��λ')
title('��λ��');

%% �Աȷ���������ֵ�����Ƶ��
W=linspace(-10,10,1000);
H=1./(1j*W+1);
%% �����׶Ա�
subplot(2,1,1);
plot(W,abs(H),w,A);
axis([-10,10,0,1])
xlabel('Ƶ��');
ylabel('����');
legend('�������','��ֵ����')
title('������')
%% ��λ�׶Ա�
subplot(2,1,2);
plot(W,angle(H),w,Z);
ylim([-2 2]);
axis([-10,10,0,1])
xlabel('Ƶ��');
ylabel('��λ');
legend('�������','��ֵ����')
title('��λ��')

%% ������⸵��Ҷ���任
syms t w;
f = 1./(w^2+1);
F=ifourier(f,t,w);
%% ���Ʋ���ͼ
t=-10:0.01:10;
f=exp(-abs(t))/2;
plot(t,f);
xlabel('t');
ylabel('y(t)');
title('����ͼ');


t = -10:0.01:10 %ʱ��Χ
w = -10:0.1:10; %Ƶ��Χ
%% 1���ź���
G=rectpuls(t,4); %G = heaviside(t+2)-heaviside(t-2) ��������ʽ������ķ����������⣩
Gf = G*exp(-1i*t'.*w)*0.01 %����ֵ��������Ҷ�任
subplot(4,3,1)
plot(t,G)
xlabel('t')
ylabel('$G_(t)$', 'Interpreter', 'latex')
title('�ź��� ����ͼ');
subplot(4,3,2)
plot(w,abs(Gf))
xlabel('Ƶ��')
ylabel('����')
title('�ź��� ������');
subplot(4,3,3)
plot(w,angle(Gf))
ylim([-5 5]);
xlabel('Ƶ��')
ylabel('��λ');
title('�ź��� ��λ��');
%% 2�����Ǻ���
t1 = -10:0.01:10
w1 =-10:0.1:10;
S = tripuls(t,4)
Sf = S*exp(-1i*t1'*w1)*0.01 %����ֵ��������Ҷ�任
subplot(4,3,4)
plot(t1,S)
xlabel('t')
ylabel('value')
title('���Ǻ��� ����ͼ');
subplot(4,3,5)
plot(w1,abs(Sf))
xlabel('Ƶ��')
ylabel('����')
title('���Ǻ��� ����ͼ');
subplot(4,3,6)
plot(w1,angle(Sf))
xlabel('Ƶ��')
ylabel('��λ')
title('���Ǻ��� ��λͼ');
%% ����ָ������
t2 = -10:0.01:10
w2 = -10:0.1:10;
X = exp(-t2).*heaviside(t2)
Xf = X*exp(-1i*t2'*w2)*0.01 %����ֵ��������Ҷ�任
subplot(4,3,7)
plot(t2,X)
xlabel('t')
ylabel('$e^{-t}\epsilon(t)$', 'Interpreter', 'latex')
title('����ָ������ ����ͼ')
subplot(4,3,8)
plot(w2,abs(Xf))
xlabel('Ƶ��')
ylabel('����')
title('����ָ������������')
subplot(4,3,9)
plot(w2,angle(Xf))
xlabel('Ƶ��')
ylabel('��λ')
title('����ָ��������λ��')
%% ���ĸ�����
t3 = -10:0.01:10
w3 = -10:0.1:10;
Y = exp(-t3).*heaviside(t3)-exp(t3).*heaviside(-t3)
Yf = Y*exp(-1i*t3'*w3)*0.01 %����ֵ��������Ҷ�任
subplot(4,3,10)
plot(t3,Y)
xlabel('t')
ylabel('$e^{-t}\epsilon(t)-e^{t}\epsilon(-t)$', 'Interpreter', 'latex')
title('���ĸ���������ͼ')
subplot(4,3,11)
plot(w3,abs(Yf))
xlabel('Ƶ��')
ylabel('����')
title('���ĸ�����������')
subplot(4,3,12)
plot(w3,angle(Yf))
xlabel('Ƶ��')
ylabel('��λ')
title('���ĸ�������λ��')



%% ����G_4(t) ���Ρ����ȡ���λͼ
t = -10:0.01:10 %ʱ��Χ
w = -10:0.1:10; %Ƶ��Χ;
G = rectpuls(t,4); %����G_4
Gf = G*exp(-1i*t'*w)*0.01 %����ֵ��������Ҷ�任
subplot(3,3,1)
plot(t,G)
xlabel('t')
ylabel('$G_4(t)$', 'Interpreter', 'latex')
title('G_4(t) ����ͼ')
subplot(3,3,2)
plot(w,abs(Gf))
ylim([0 8]) % ���������ķ��������8��ֱ����������ȷ�Χ������Ϊ[0 8]
xlabel('Ƶ��')
ylabel('����')
title('G_4(t) ������')
subplot(3,3,3)
plot(w,angle(Gf))
ylim([-5 5]) % ��λ��Χ������Ϊ[-5 5]
xlabel('Ƶ��')
ylabel('��λ')
title('G_4(t) ��λͼ')
%% ����G_4(t/2) ���Ρ����ȡ���λͼ
t1 = -10:0.01:10 %ʱ��Χ
w1 = -10:0.1:10; %Ƶ��Χ
G1 = rectpuls(t1/2,4)
Gf1 = G1*exp(-1i*t1'*w1)*0.01
subplot(3,3,4)
plot(t1,G1)
xlabel('t')
ylabel('$G_4(t/2)$', 'Interpreter', 'latex')
title('G_4(t/2) ����ͼ')
subplot(3,3,5)
plot(w1,abs(Gf1))
ylim([0 8])
xlabel('Ƶ��')
ylabel('����')
title('G_4(t/2) ������')
subplot(3,3,6)
plot(w1,angle(Gf1))
ylim([-5 5])
xlabel('Ƶ��')
ylabel('��λ')
title('G_4(t/2) ��λ��')
%% ����G_4(2t) ���Ρ����ȡ���λͼ
t2=-10:0.01:10
w2 =-10:0.1:10;
G2=rectpuls(t2*2,4);
Gf2=G2*exp(-1i*t2'*w2)*0.01
subplot(3,3,7)
plot(t2,G2)
xlabel('t')
ylabel('$G_4(2t)$', 'Interpreter', 'latex')
title('G_4(2t) ����ͼ')
subplot(3,3,8)
plot(w2,abs(Gf2))
ylim([0 8])
xlabel('Ƶ��')
ylabel('����')
title('G_4(2t) ������')
subplot(3,3,9)
plot(w2,angle(Gf2))
ylim([-5 5])
xlabel('Ƶ��')
ylabel('��λ')
title('G_4(2t) ��λ��')


%% ����G_4(t) ���Ρ����ȡ���λͼ
t1= -10:0.01:10 %ʱ��Χ
w1 = -10:0.01:10;%Ƶ��Χ
S =( t1+2).*(heaviside(t1+2)-heaviside(t1)).*0.5+(-t1+2).*(heaviside(t1)-heaviside(t1-2)).*0.5 %���庯��
Sf = S*exp(-1i*t1'*w1)*0.01 %��ֵ�������㸵��Ҷ�任
subplot(3,3,1)
plot(t1,S)
xlabel('t')
ylabel('value')
title('���Ǻ��� ����ͼ')
subplot(3,3,2)
plot(w1,abs(Sf))
xlabel('Ƶ��')
ylabel('����')
title('���Ǻ���������')
subplot(3,3,3)
plot(w1,angle(Sf))
ylim([-1e-10,1e-10])
xlabel('Ƶ��')
ylabel('��λ')
title('���Ǻ��� ��λ��')

%% ����?_4(t-0.1) ���Ρ����ȡ���λͼ
t2 = -10:0.01:10 %ʱ��Χ
w2 = -10:0.01:10; %Ƶ��Χ
S2 = (t2+2-0.1).*(heaviside(t2-0.1+2)-heaviside(t2-0.1)).*0.5+(-t2+2+0.1).*(heaviside(t2-0.1)-heaviside(t2-2-0.1)).*0.5 %���庯��
Sf2 = S2*exp(-1i*t2'*w2)*0.01 %��ֵ�������㸵��Ҷ�任
subplot(3,3,4)
plot(t2,S2)
xlabel('t')
ylabel('value')
title('���Ǻ�������0.1 ����ͼ')
subplot(3,3,5)
plot(w2,abs(Sf2))
xlabel('Ƶ��')
ylabel('����')
title('���Ǻ�������0.1 ������')
subplot(3,3,6)
plot(w2,angle(Sf2))
xlabel('Ƶ��')
ylabel('��λ')
title('���Ǻ�������0.1 ��λ��')

%% ����G_4(t-1) ���Ρ����ȡ���λͼ
t3=-10:0.01:10  %ʱ��Χ
w3 =-10:0.01:10; %Ƶ��Χ
S3=(t3-1+2).*(heaviside(t3-1+2)-heaviside(t3-1)).*0.5+(-t3+1+2).*(heaviside(t3-1)-heaviside(t3-1-2)).*0.5 %���庯��
Sf3=S3*exp(-1i*t3'*w3)*0.01 %��ֵ�������㸵��Ҷ�任
subplot(3,3,7)
plot(t3,S3)
xlabel('t')
ylabel('value')
title('���Ǻ�������1 ����ͼ')
subplot(3,3,8)
plot(w3,abs(Sf3))
xlabel('Ƶ��')
ylabel('����')
title('���Ǻ�������1 ������')
subplot(3,3,9)
plot(w3,angle(Sf3))
xlabel('Ƶ��')
ylabel('��λ')
title('���Ǻ�������1 ��λ��')


%% ����G_4(t) ���Ρ����ȡ���λͼ
t = -5:0.01:5%ʱ��Χ
w = -10:0.01:10;%Ƶ��Χ
G = rectpuls(t,4); %���庯��
Gf = G*exp(-1i*t'*w)*0.01 %��ֵ�������㸵��Ҷ�任
subplot(2,3,1)
plot(t,G)
xlabel('t')
ylabel('$G_4(t)$', 'Interpreter', 'latex')
title('G_4(t) ����ͼ')
subplot(2,3,2)
plot(w,abs(Gf))
xlabel('Ƶ��')
ylabel('����')
title('G_4(t) ������')
subplot(2,3,3)
plot(w,angle(Gf))
xlabel('Ƶ��')
ylabel('��λ')
title('G_4(t) ��λ��')
%% ����G_4(t)cos(20t) ���Ρ����ȡ���λͼ
t1 = -5:0.01:5 %ʱ��Χ
w1 = -10:0.01:10; %Ƶ��Χ
G1 = rectpuls(t,4).*cos(20*t1) %���庯��
Gf1 = G1*exp(-1i*t1'*w1)*0.01 %��ֵ�������㸵��Ҷ�任
subplot(2,3,4)
plot(t1,G1)
xlabel('t')
ylabel('$G_4(t)cos(20t)$', 'Interpreter', 'latex')
title('G_4(t)cos(20t) ����ͼ')
subplot(2,3,5)
plot(w1,abs(Gf1))
xlabel('Ƶ��')
ylabel('����')
title('G_4(t)cos(20t) ������')
subplot(2,3,6)
plot(w1,angle(Gf1))
xlabel('Ƶ��')
ylabel('��λ')
title('G_4(t)cos(20t) ��λ��')
