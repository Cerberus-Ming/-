%% t��ȡֵ��ΧΪ0-2��ÿ��0.01��һ��ȡֵ��
t = 0:0.01:2;
%% ������ʽ��heavisideΪ��Ծ������ע����
x = sin(2*pi*t) .* (heaviside(t) - heaviside(t-4));
%% ��ͼ
plot(t,x);
%% ����y�᷶Χ��������ʾ
ylim([-1.1 1.1]);
%% ����x y�ᣬͼ������
xlabel('t');
ylabel('x(t)');
title('$x(t) = \sin(2\pi t)[u(t) - u(t-4)]$', 'Interpreter', 'latex');

%% t��ȡֵ��ΧΪ0-2��ÿ��0.01��һ��ȡֵ��
t =0:0.01:2;
%% ������ʽ��heavisideΪ��Ծ������expΪָ��������ע����
h = exp(-t) .* heaviside(t) - exp(-2*t) .* heaviside(t);
%% ��ͼ
plot(t,h);
%% ����y�᷶Χ��������ʾ
ylim([0 0.28]);
%% ����x y�ᣬͼ������
xlabel('t');
ylabel('h(t)');
title('$h(t) = \exp(-t)u(t) - \exp(-2t)u(t)$', 'Interpreter', 'latex');

%% t��ȡֵ��ΧΪ-4-+4��ÿ��0.01��һ��ȡֵ��
t = -4:0.01:4;
%% �ý�Ծ����ʵ���ź���
y = 2*heaviside(t+2) - 2*heaviside(t-2);
%% ��ͼ
plot(t,y);
%% ����y�᷶Χ��������ʾ
ylim([0 2.1]);
%% ����x y�ᣬͼ������
xlabel('t');
ylabel('y(t)');
title('$y(t) = 2u(t+2) - 2u(t-2)$', 'Interpreter', 'latex');


%% t��ȡֵ��ΧΪ-2-+2��ÿ��0.01��һ��ȡֵ��
t = -2:0.01:2;
%% ���庯��
G1 = heaviside(t+0.5) - heaviside(t-0.5);%G1(t)
y1 = heaviside(2*t+0.5) - heaviside(2*t-0.5);%y(2t)
y2 = heaviside(t/2+0.5) - heaviside(t/2-0.5);%y(t/2)
y3 = heaviside((2-2*t)+0.5) - heaviside((2-2*t)-0.5);%y(2-2t)
%% ����y(t),y(2t)
%% subplot(m, n, p) ��ͼ�δ��ڷֳ� m �� n �е���ͼ����,��ǰ��ͼΪ�� p ����ͼ
subplot(3,1,1);
plot(t,G1,t,y1);
ylim([0 1.1]);
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('y(t)');
legend('y(t)','y(2t)');
title('ͼ1 y(t),y(2t)');
%% ����y(t),y(t/2)
subplot(3,1,2);
plot(t,G1,t,y2);
ylim([0 1.1]);
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('y(t)');
legend('y(t)','y(t/2)');
title('ͼ2 y(t),y(t/2)');
%% ����y(t),y(2-2t)
subplot(3,1,3);
plot(t,G1,t,y3);
ylim([0 1.1]);
%% ����xy�����ᣬ��ͼ����
xlabel('t');
ylabel('y(t)');
legend('y(t),y(2-2t)');
title('ͼ3 y(t),y(2-2t)');


%% t��ȡֵ��ΧΪ-100-+100��ÿ��0.01��һ��ȡֵ��
t = -100:0.01:100;
%% ���庯��
y = cos(t) + cos(pi*t/4);
%% ���ƺ���
plot(t,y);
%% ���������ᣬͼ������
xlabel('t');
ylabel('y(t)');
title( '$y(t) = \cos(t) + \cos(\pi t/4 )$', 'Interpreter', 'latex');

%% t��ȡֵ��ΧΪ-10-+10��ÿ��0.01��һ��ȡֵ��
t = -10:0.01:10;
%% ���庯��
y = sin(pi*t) + cos(2*pi*t);
%% ���ƺ���
plot(t,y);
%% ���������ᣬͼ������
xlabel('t');
ylabel('y(t)');
title( '$y(t) = \sin(t) + \cos(2\pi t )$', 'Interpreter', 'latex');


t = 0:0.01:20;
y1 = exp(-2 * t) .* heaviside(t);
y2 = exp(-t) .* heaviside(t);
%% ʹ��conv��y1��y2���о��
y = conv(y1, y2) .* 0.01;%���ڼ�������ɢ�ĵ㣬�������Ҫ���Բ���
k = 2*length(t)-1;
k1 = linspace(2*t(1),2*t(end),k);
%% ����conv������õ�ͼ��
subplot(3,1,1);
plot(k1, y);
xlabel('t');
ylabel('y(t)');
title('ͼ1 ��ֵ������')
legend('��ֵ');
%% ��������ֵͼ��
t_theory = 0:0.01:40;
y_theory = (exp(-t_theory) - exp(-2*t_theory));
subplot(3,1,2);
plot(t_theory, y_theory);
xlabel('t');
ylabel('y(t)');
title('ͼ2 �����Ƶ����');
legend('����');
%% ������ֵ������2�е���ֵ����������һ��ͼ��
subplot(3,1,3);
plot(k1,y,t_theory,y_theory);
xlabel('t');
ylabel('y(t)');
title('ͼ3 ��ֵ����������Ƶ��Ա�');
legend('��ֵ','����');%��legend����ͼ��


t=0:0.01:20
%% ����ʱ��LTIϵͳH����ͨ��tf(b, a)����
%% b��a�ֱ�Ϊ΢�ַ����Ҷ˺���˸����ϵ������.
H=tf([1],[1 3 2]);
e=exp(-2*t).*heaviside(t);%���������ź�
y=lsim(H,e,t);%ʹ��lsim���������״̬��Ӧ
%% ������ֵ����ͼ��
subplot(3,1,1);
plot(t,y);
xlabel('t');
ylabel('r_{zs}');
title('ͼ1 ��ֵ������');
legend('��ֵ');
%% ���������Ƶ�ͼ��
y1 = exp(-t) - (1+t).*exp(-2.*t);
subplot(3,1,2);
plot(t,y1);
xlabel('t');
ylabel('r_{zs}');
title('ͼ2 �����Ƶ����');
legend('����');
%% ���ۺͷ���Ա�
subplot(3,1,3);
plot(t,y,t,y1);
xlabel('t');
ylabel('r_{zs}');
title('ͼ3 ��ֵ����������Ƶ��Ա�');
legend('��ֵ','����')