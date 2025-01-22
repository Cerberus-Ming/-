import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;

/**
 * @description: Server 类用于启动 RMI 服务器，创建远程对象并注册到 RMI Registry 中
 * @author Eternity
 * @date 2023/11/24 14:21
 * @version 1.0
 */
public class Server {

    public static void main(String[] args) {
        try {
            // 创建 RMI 注册表，使用默认端口 1099
            LocateRegistry.createRegistry(1099);

            // 创建 InterfaceImplements 类的实例作为远程对象
            MessageInterface messageInterface = new InterfaceImplements();

            // 将远程对象注册到 RMI 注册表中，命名为 "Message"
            Naming.rebind("Message", messageInterface);

            System.out.println("服务器启动成功！");
        } catch (RemoteException | MalformedURLException e) {
            e.printStackTrace();
        }
    }
}
