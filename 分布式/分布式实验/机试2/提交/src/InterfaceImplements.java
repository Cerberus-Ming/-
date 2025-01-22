import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.Date;

/**
 * @description: 该类用于实现服务器端的远程方法调用（RMI）功能。
 * @author Eternity
 * @date 2023/11/24 14:32
 * @version 1.0
 */
public class InterfaceImplements extends UnicastRemoteObject implements MessageInterface {

    /**
     * 存储所有消息
     */
    ArrayList<Message> messages = new ArrayList<Message>();

    /**
     * 存储所有用户
     */
    ArrayList<User> users = new ArrayList<User>();

    int ID = 0;

    /**
     * InterfaceImplements 构造方法
     *
     * @throws RuntimeException
     * @throws RemoteException
     */
    public InterfaceImplements() throws RuntimeException, RemoteException {
        super();
    }

    /**
     * 判断用户名为 username，密码为 password 的用户是否存在。
     * @param username
     * @param password
     * @return
     */
    public boolean userExist(String username, String password) {
        for (User user : users) {
            if (user.getUsername().equals(username) && user.getPassword().equals(password)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 获取用户名为 username 的用户。
     * @param username
     * @return
     */
    public User getUser(String username) {
        for (User user : users) {
            if (user.getUsername().equals(username)) {
                return user;
            }
        }
        return null;
    }

    /**
     * 新用户注册。如果用户名已存在，则提示用户选择另一个用户名。
     * @param username
     * @param password
     * @return
     * @throws RemoteException
     */
    @Override
    public boolean register(String username, String password) throws RemoteException {
        //首先判断该用户是否已经注册
        if (getUser(username) == null) {
            User user = new User(username, password);
            users.add(user);
            return true;
        } else {
            System.out.println(username + "已经被注册，请换一个用户名!");
            return false; // 返回false，便于客户端进行判断
        }
    }

    /**
     * 显示所有注册的用户
     * @return
     * @throws RemoteException
     */
    @Override
    public ArrayList<String> showUsers() throws RemoteException {
        ArrayList<String> userShow = new ArrayList<String>();
        for (User user : users) {
            userShow.add(user.getUsername());
        }
        return userShow;
    }

    /**
     * 返回指定用户的所有留言
     * 如果用户名和密码不对，则有相应的提示信息；
     * 如果该用户没有任何留言，也应该有提示。
     * @param username
     * @param password
     * @return
     * @throws RemoteException
     */
    @Override
    public ArrayList<Message> checkMessages(String username, String password) throws RemoteException {
        // 输入条件不正确
        if (!userExist(username, password)) {
            System.err.println("用户名或密码输入错误！");
            return null;
        }
        // 返回指定用户的所有留言
        ArrayList<Message> messages1 = new ArrayList<Message>();
        for (Message message : messages) {
            if (message.getReceiver().getUsername().equals(username)) {
                messages1.add(message);
            }
        }
        return messages1;
    }

    /**
     * 给其他用户留言
     * 若信息不正确，请给出相应的提示，留言不成功；
     * 若信息均正确，则系统记录留言的日期和时间、留言内容等信息。
     *
     * @param username
     * @param password
     * @param receiverName
     * @param message
     * @return
     * @throws RemoteException
     */
    @Override
    public int leaveMessage(String username, String password, String receiverName, String message) throws RemoteException {
        // 校验用户名和密码是否正确
        if (!userExist(username, password)) {
            System.err.println("用户名或密码输入错误！");
            return -1;
        } else {
            //校验接收者用户名是否存在
            User sender = getUser(username);
            User receiver = getUser(receiverName);
            if (receiver == null) {
                System.err.println("接收者不存在");
                return 0;
            }
            Message msg = new Message(ID++, new Date(), message, sender, receiver);
            messages.add(msg);
            return 1;
        }
    }
}
