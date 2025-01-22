/**
 * @Author shenming
 * @Date 2023 10 13 10 00
 **/

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

/**
 * @description: 客户端
 * @author Eternity
 * @date 2023/10/03 10:00
 * @version 1.0
 */
public class FileClient {
    static final int tcp_prot = 2021;   //TCP端口号
    static final int udp_port = 2020;   //UDP端口号
    static final String host = "127.0.0.1";  //IP地址(127.0.0.1: 代表本机IP)
    public int packet_size = 8*1024;   //UDP数据包大小(小于64KB)

    //下面是IO相关变量
    public BufferedReader bufferedReader = null;    //用于接受服务器的信息(将InputStream转化成比较好用的BufferedReader)
    public BufferedWriter bufferedWriter = null;    //用于接受服务器的信息(将OutputStream转化成比较好用的BufferedWriter)
    public PrintWriter printWriter = null;  //用于输出
    public Scanner scanner= null;   //用于输入

    Socket socket = null;   //客户端socket

    /**
     * 构造器
     * @throws IOException
     */
    public FileClient() throws IOException {
        //创建客户端socket
        socket = new Socket();
        //与服务端TCP端口建立连接
        socket.connect(new InetSocketAddress(host, tcp_prot));
        //初始化所有IO相关的变量
        bufferedWriter = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
        bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        printWriter = new PrintWriter(bufferedWriter, true);
        scanner = new Scanner(System.in);

        System.out.println("成功连接服务端。");
    }

    /**
     * 关闭IO
     * 在关闭TCP连接的时候，关闭所有IO流，释放资源
     */
    public void closeIO(){
        try {
            bufferedWriter.close();
            bufferedReader.close();
            printWriter.close();
            scanner.close();
            System.out.println("IO关闭");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 通过TCP给服务端发送命令
     * 先对用户输入的指令进行初步检验，判断指令正确与否，再将指令发送至服务器端
     * 再对接受到的数据，调用不同函数进行后续处理
     * 命令格式：
     * [1]	ls	服务器返回当前目录文件列表（<file/dir>	name	size）
     * [2]	cd  <dir>	进入指定目录（需判断目录是否存在，并给出提示）
     * [3]	get <file>	通过UDP下载指定文件，保存到客户端当前目录下
     * [4]	bye	断开连接，客户端运行完毕
     * @throws IOException
     */
    public void run() throws IOException {
        String msg = null;   //存储用户输入
        Boolean bye = false;    //输入是否是bye

        while(true){
            msg = scanner.nextLine();   //读入一行
            //如果是空行，就进入下一次循环，再次读取
            if(msg == null){
                continue;
            }

            //1、给服务器发消息
            printWriter.println(msg);
            //2、发送完消息的操作
            //2.1、从msg中提取cmd（操作）
            String cmd = msg.split(" ")[0];
            //2.2、根据cmd，进行不同的操作
            switch(cmd){
                //bye, 设置bye变量为true，后面退出循环
                case "bye":{
                    bye = true;
                    break;
                }
                case "ls":{
                    ls();
                    break;
                }
                case "cd":{
                    //如果msg符合规范（cd *目录*），那么调用cd函数
                    if(msg.split(" ").length == 2){
                        cd();
                    }else{
                        System.out.println(bufferedReader.readLine());
                    }
                }
            }
        }
    }

    /**
     * ls命令的后续操作：
     * 输出当前目录的所有文件
     */
    public void ls() throws IOException {
        String out = bufferedReader.readLine();
        while(!"".equals(out)) {
            System.out.println(out);
            out = bufferedReader.readLine();
        }
    }

    /**
     * cd命令的后续操作：
     * 输出当前目录
     */
    public void cd() throws IOException {
        System.out.println(bufferedReader.readLine());
    }

    /**
     * get命令的操作：
     * 在UDP连接上传输文件
     * @param path
     * @throws IOException
     */
    public void get(String path) throws IOException{
        String out;
        String downloadPath = new File("").getAbsolutePath();
        String[] tokens;
        String fileName;
        BufferedOutputStream bos = null;

        try{
            tokens = path.split("/");
            fileName = tokens[tokens.length - 1];
        }catch (Exception e){
            e.printStackTrace();
            return;
        }

        if (!"".equals(out = bufferedReader.readLine())){
            // 出现异常，退出get接收
            System.out.println(out);
            return;
        }

        File file = new File(downloadPath + "/" + fileName);
        if (file.createNewFile()){
            // true表示创建成功，false表示文件已存在
            System.out.println("had create a new file " + fileName);
        }

        // 开始udp传输================================

        DatagramPacket send, recv;
        DatagramSocket datagramSocket = new DatagramSocket();
        byte[] sendBuf, recvBuf;
        InetAddress loaclhost = InetAddress.getByName(host);
        try {
            while (true) {

                int len = "start".getBytes(StandardCharsets.UTF_8).length;
                send = new DatagramPacket("start".getBytes(StandardCharsets.UTF_8), len, loaclhost, udp_port);
                datagramSocket.send(send);

                len = "trans".getBytes(StandardCharsets.UTF_8).length;
                recvBuf = new byte[len];
                recv = new DatagramPacket(recvBuf, len);
                datagramSocket.receive(recv);

                String tmp = new String(recv.getData(), StandardCharsets.UTF_8);
                System.out.println(tmp);
                if ("trans".equals(tmp)) {

                    break;
                }
            }

            System.out.println("准备就绪");

            // 接收文件
            bos = new BufferedOutputStream(new FileOutputStream(file));
            recvBuf = new byte[packet_size];
            while (true) {
                recv = new DatagramPacket(recvBuf, packet_size);
                datagramSocket.receive(recv);

                String tmp = new String(recv.getData(), 0, recv.getLength());
                if ("end".equals(tmp)) {
                    break;
                }

                bos.write(recv.getData(), 0, recv.getLength());
                bos.flush();
                byte[] ok = "OK".getBytes(StandardCharsets.UTF_8);
                send = new DatagramPacket(ok, ok.length, loaclhost, udp_port);
                datagramSocket.send(send);

            }

            System.out.println("接收成功");
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            bos.close();
            datagramSocket.close();

        }

    }

    /**
     * 主函数
     * @param args
     * @throws UnknownHostException
     * @throws IOException
     */
    public static void main(String[] args) throws UnknownHostException, IOException{
        FileClient fileClient = new FileClient();
        fileClient.run();
    }
}
