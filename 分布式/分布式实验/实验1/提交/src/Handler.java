/**
 * @Author shenming
 * @Date 2023 10 20 14 37
 **/

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

import static java.nio.file.Paths.get;

/**
 * @description: Handler类实现服务器线程
 * @author Eternity
 * @date 2023/10/04 14:37
 * @version 1.0
 */
public class Handler implements Runnable{

    static final int udp_port = 2020;   //UDP端口号
    int packet_size = 8*1024;   //每次传输的UDP数据包的大小

    //下面是IO相关的变量
    public BufferedReader bufferedReader = null;    //用于接受服务器的信息(将InputStream转化成比较好用的BufferedReader)
    public BufferedWriter bufferedWriter = null;    //用于接受服务器的信息(将OutputStream转化成比较好用的BufferedWriter)
    public PrintWriter printWriter = null;  //用于输出

    private Socket socket;  //对应客户端的socket

    DatagramSocket datagramSocket = null;   //UDP服务端

    String path = null; //当前目录

    /**
     * 构造函数
     * @param socket
     * @param path
     */
    public Handler(Socket socket, String path){
        this.socket = socket;
        this.path = path;
    }

    /**
     * 初始化所有IO流
     */
    public void initStream() throws IOException {

        // 初始化输入输出流对象方法
        bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        bufferedWriter = new BufferedWriter(
                new OutputStreamWriter(socket.getOutputStream()));
        printWriter = new PrintWriter(bufferedWriter, true);
    }
    /**
     * 关闭所有IO流
     */
    public void close(){
        try{
            bufferedWriter.close();
            bufferedReader.close();
            printWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void run() {

        Boolean bye = false;    //bye命令标志
        //客户端信息
        String connectInfo = socket.getInetAddress() + ":" + socket.getPort();

        //提示有新连接
        System.out.println("新连接——地址：" + connectInfo);
        try {
            initStream(); //初始化IO流
            String msg = null;
            while((msg = bufferedReader.readLine()) != null){
                //提取命令
                String [] orders = msg.split(" ");
                String cmd = orders[0];

                switch (cmd){
                    case "bye":
                        //断开连接，通过设置bye为true结束run循环
                        System.out.println(connectInfo + " Disconnect.");
                        bye = true;
                        break;
                    case "ls":
                        //返回当前目录文件列表（<file/dir>	name size）
                        System.out.println(connectInfo + " ls");
                        ls(this.path);
                        break;
                    case "cd..":
                        //退回上一级文件目录
                        System.out.println(connectInfo + " cd..");
                        cd("..");
                        break;
                    case "cd":
                        //判断格式是否符合：cd  <dir>
                        if(orders.length == 2){

                            System.out.println(connectInfo + " cd " + this.path);
                            cd(orders[1]);
                        }else {
                            printWriter.println("You need input: 'cd  <dir>'.");
                        }
                        break;
                    case "get":
                        //判断格式是否符合：get  <file>
                        if(orders.length == 2){
                            System.out.println(connectInfo + " get " + orders[1]);
                            String downloadPath = bufferedReader.readLine();
                            get(orders[1], downloadPath);
                        }else {
                            printWriter.println("You need input: 'get <file>'.");
                        }
                        break;
                    default:
                        printWriter.println("Unknown input!");
                        break;
                }
                //如果输入的是bye，则退出
                if(bye){
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            //关闭连接
            if(socket != null){
                try {
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            //关闭IO
            close();
        }
    }

    /**
     * 展示当前目录下的所有文件
     * @param path 当前目录路径
     */
    public void ls(String path){
        String out = "";    //输出字符串
        File file = new File(path); //当前目录文件
        // 检验目录的正确性
        if(file.exists()){
            //当前目录空
            if(file.listFiles() == null){
                printWriter.println("Empty directory!");
                return;
            }
            //当前目录不为空，则挨个输出目录或文件
            for(File f : file.listFiles()){
                if(f.isDirectory()){
                    out = "<dir>\t" + f.getName() + "\t" + f.length();
                }else if(f.isFile()){
                    out = "<file>\t" + f.getName() + "\t" + f.length();
                }
                printWriter.println(out);   //使用TCP发送
            }
        }
        //如果当前目录不正确
        else{
            printWriter.println(path + "does ont exist!");
        }
        printWriter.println("");    //防止null
    }

    /**
     * 改变当前目录
     * @param path 绝对路径 || 相对路径 || ..
     */
    public void cd(String path){
        String t;
        //cd .. 的情况
        if("..".equals(path)){
            t = new File(this.path).getParent();
            //当前目录已经是根目录
            if(path == "D:/courseEnv" || t == null){
                printWriter.println("已经是根目录！");
                return;
            }
            //当前目录不是根目录
            this.path = t;
        }
        //cd 绝对路径 || cd 相对路径 的情况
        else {
            File file = new File(path);
            if (file.exists()) {
                //绝对路径
                this.path = file.getAbsolutePath();
            } else {
                //相对路径
                t = this.path;
                for (String token : path.split("/")) {
                    t += "/" + token;
                    file = new File(t);
                    if (!file.exists()) {
                        // 如果目录是不存在，立即返回
                        printWriter.println(path + " does not exist!");
                        return;
                    }
                }
                //如果中间路径都有效，最后将当前路径path设置绝对路径
                this.path = file.getAbsolutePath();
            }
        }
        printWriter.println(this.path + " > OK");
    }

    /**
     * 下载文件
     * @param path 要下载的文件名
     * @param downloadpath  下载后存储的目录
     */
    public void get(String path, String downloadpath){

        BufferedInputStream bufferedInputStream = null;

        try {
            //开启UDP端口监听
            datagramSocket = new DatagramSocket(udp_port);

            //合成要下载的文件的绝对路径
            System.out.println(this.path + "/" + path);
            File file = new File(this.path + "/" + path);

            //文件不存在的情况
            if(!file.exists()){
                printWriter.println("path" + "does not exist!");
                return;
            }

            //输入的不是文件名
            if(file.isDirectory()){
                printWriter.println(path + "is not a file!");
                return;
            }

            //输入的是文件名
            printWriter.println("");

            //--------开始UDP传输----------

            DatagramPacket send, recv;
            byte[] sendBuf, recvBuf;
            InetAddress remote = null;
            int port = 0;

            while (true) {

                int len = "start".getBytes(StandardCharsets.UTF_8).length;
                recvBuf = new byte[len];
                recv = new DatagramPacket(recvBuf, len);

                datagramSocket.receive(recv);
                String tmp = new String(recv.getData(), StandardCharsets.UTF_8);
                //            System.out.println(tmp);
                if ("start".equals(tmp)) {

                    remote = recv.getAddress();
                    port = recv.getPort();

                    len = "trans".getBytes(StandardCharsets.UTF_8).length;
                    send = new DatagramPacket("trans".getBytes(StandardCharsets.UTF_8), len, remote, port);
                    datagramSocket.send(send);

                    break;
                }
            }

            System.out.println("准备就绪: " + remote + ":" + port);

            // 传输文件
            bufferedInputStream = new BufferedInputStream(new FileInputStream(file));
            int readLen, k = 0;
            sendBuf = new byte[packet_size];
            while ((readLen = bufferedInputStream.read(sendBuf)) != -1) {
                send = new DatagramPacket(sendBuf, readLen, remote, port);
                datagramSocket.send(send);

                // 等待回复
                datagramSocket.receive(recv);
            }

            byte[] end = "end".getBytes(StandardCharsets.UTF_8);
            send = new DatagramPacket(end, end.length, remote, port);
            datagramSocket.send(send);

            System.out.println("传输完成");

        } catch (SocketException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            //关闭IO流
            if (bufferedInputStream != null){
                try {
                    bufferedInputStream.close();
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
            //关闭UDP连接
            datagramSocket.close();
        }
    }
}
