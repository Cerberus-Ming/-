import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @description: 服务端
 * @author Eternity
 * @date 2023/10/04 15:36
 * @version 1.0
 */
public class FileServer {

    static final int TCP_PORT = 2021;   //TCP端口号
    final int POOL_SIZE = 4;    //单个处理器线程池工作线程数目
    private static File rootFile;   //根目录
    private ServerSocket serverSocket = null;   //服务端套接字，监听TCP连接请求
    private ExecutorService executorService = null; //线程池

    /**
     * 构造函数
     * @throws IOException
     */
    public FileServer() throws IOException{
        //为服务端创建一个ServerSocket对象，同时为服务端注册端口TCP_Port
        serverSocket = new ServerSocket(TCP_PORT);
        //创建线程池
        //线程池的大小为当前计算机上可用处理器核心数量乘以 POOL_SIZE
        executorService = Executors.newFixedThreadPool(Runtime.getRuntime()
                .availableProcessors() * POOL_SIZE);
        System.out.println("服务器启动。");
    }

    /**
     * 循环接收tcp请求，分配线程，开启服务
     * @param path 初始目录
     */
    public void service(String path){
        //服务端循环等待新的客户端TCP会话
        while (true){
            try{
                //调用serverSocket对象的accept方法，等待客户端的连接请求
                Socket socket = serverSocket.accept();
                //每收到一个客户端的连接请求时，就把这个客户端对应的socket通信通道交给一个独立的线程处理
                //执行由线程池维护
                executorService.execute(new Handler(socket, path));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    /**
     * 主函数
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 1){
            System.out.println("usage: java FileServer <dir>.");    //使用说明
            return;
        }

        rootFile = new File(args[0]);

        if (!rootFile.exists() || !rootFile.isDirectory()){
            // 文件夹不存在 || 不是一个文件夹 的情况
            System.out.println(rootFile.getAbsoluteFile() +
                    " 输入的不是合法的路径");
            return;
        }

        FileServer fileServer = new FileServer();
        fileServer.service(args[0]);
    }
}
