import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.util.ArrayList;


/**
 * @description: Client 类负责通过 RMI（远程方法调用）与远程服务器进行交互
 * @author Eternity
 * @date 2023/11/24 14:23
 * @version 1.0
 */
public class Client {

    public static void main(String[] args) {

        String hostIP; // 主机IP
        int port;  // 端口名
        BufferedReader keyboard = new BufferedReader(new InputStreamReader(System.in));

        //程序实参：主机IP和端口名 建议：localhost 1099
        if (args.length != 2) {
            System.err.println("参数传递错误！");
            return;
        } else {
            hostIP = args[0];
            port = Integer.parseInt(args[1]);
        }
        // 初始化用于 RMI 通信的 MessageInterface
        MessageInterface messageInterface = null;


        try {
            // 使用 RMI 命名查找远程对象
            messageInterface = (MessageInterface) Naming.lookup("rmi://" + hostIP + ":" + port + "/Message");
        } catch (NotBoundException | MalformedURLException | RemoteException e) {
            e.printStackTrace();
        }

        int choice = 0;

        // 用户交互的主循环
        while (true) {
            menu();
            try {
                // 读取用户输入
                choice = Integer.parseInt(keyboard.readLine());
            } catch (IOException e) {
                e.printStackTrace();
            }

            String username, password, receiverName, message;
            switch (choice) {
                case 1: {
                    //register
                    try {
                        System.out.println("请输入用户名：");
                        username = keyboard.readLine();
                        System.out.println("请输入密码：");
                        password = keyboard.readLine();

                        assert messageInterface != null;
                        // 在远程服务器上调用 register 方法
                        if (!messageInterface.register(username, password)) {
                            System.err.println("注册失败，该用户名已被占用！");
                        } else {
                            System.out.println("注册成功！");
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    break;
                }
                case 2: {
                    //showUsers
                    try {
                        assert messageInterface != null;
                        // 在远程服务器上调用 showUsers 方法
                        ArrayList<String> users = messageInterface.showUsers();
                        System.out.println("---------用户列表---------");
                        for (String user : users) {
                            System.out.println(user);
                        }
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }
                    break;
                }
                case 3: {
                    //checkMessages
                    try {
                        System.out.println("请输入用户名：");
                        username = keyboard.readLine();
                        System.out.println("请输入密码：");
                        password = keyboard.readLine();
                        assert messageInterface != null;
                        // 在远程服务器上调用 checkMessages 方法
                        ArrayList<Message> messages = messageInterface.checkMessages(username, password);

                        if (messages == null) {
                            System.err.println("用户名或密码输入错误");
                        } else {
                            if (messages.size() == 0) {
                                System.out.println("您没有任何留言！");
                            } else {
                                System.out.println("----------留言列表----------");
                                for (Message info : messages) {
                                    System.out.println(info);
                                }
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    break;
                }
                case 4: {
                    //leaveMessage
                    try {
                        // 读取消息相关信息
                        System.out.println("请输入用户名：");
                        username = keyboard.readLine();
                        System.out.println("请输入密码：");
                        password = keyboard.readLine();
                        System.out.println("请输入接收用户名：");
                        receiverName = keyboard.readLine();
                        System.out.println("请输入消息：");
                        message = keyboard.readLine();

                        assert messageInterface != null;
                        // 在远程服务器上调用 leaveMessage 方法
                        int result = messageInterface.leaveMessage(username,password,receiverName,message);
                        // 根据返回值进行后续操作
                        if(result == -1) {
                            System.err.println("用户名或密码错误！");
                        } else if (result == 0) {
                            System.err.println("接收者不存在！");
                        } else if (result == 1) {
                            System.out.println("留言成功！");
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    break;
                }
                case 5:{
                    System.out.println("再见！");
                    System.exit(0);
                }
                default :{
                    // 输入不在1-5内的数字 输出提示信息
                    System.err.println("选项输入错误！请输入1-5");
                }
            }
        }
    }

    /**
     * 显示菜单
     */
    private static void menu() {
        System.out.println("菜单：");
        System.out.println("[1] register(username, password)");
        System.out.println("[2] showusers()");
        System.out.println("[3] checkmessages(username, password)");
        System.out.println("[4] leavemessage(username, password, receiver_name, message_text)");
        System.out.println("[5] exit");
        System.out.println("请输入您的选择：");
    }
}