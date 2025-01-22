import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Objects;

/**
 * @description: 消息对象类
 * @author Eternity
 * @date 2023/11/24 14:26
 * @version 1.0
 */
public class Message implements Serializable {

    private int ID; // 消息的ID

    private Date messageDate;   // 消息的日期

    private String message; // 消息的内容

    private User sender;   // 消息发送方

    private User receiver;  // 消息接收方

    private final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    /**
     * 构造函数
     * @param ID
     * @param messageDate
     * @param message
     * @param sender
     * @param receiver
     */
    public Message(int ID, Date messageDate, String message, User sender, User receiver) {
        this.ID = ID;
        this.messageDate = messageDate;
        this.message = message;
        this.sender = sender;
        this.receiver = receiver;
    }

    public int getId() {
        return ID;
    }

    public void setId(int id) {
        this.ID = id;
    }

    public Date getMessageDate() {
        return messageDate;
    }

    public void setMessageDate(Date messageDate) {
        this.messageDate = messageDate;
    }

    public User getSender() {
        return sender;
    }

    public void setSender(User sender) {
        this.sender = sender;
    }

    public User getReceiver() {
        return receiver;
    }

    public void setReceiver(User receiver) {
        this.receiver = receiver;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Message message1 = (Message) o;
        return ID == message1.ID && Objects.equals(messageDate, message1.messageDate) && Objects.equals(message, message1.message) && Objects.equals(sender, message1.sender) && Objects.equals(receiver, message1.receiver) && Objects.equals(sdf, message1.sdf);
    }

    /**
     * 重写 hashCode 方法，用于获取消息对象的哈希码
     * @return
     */
    @Override
    public int hashCode() {
        return Objects.hash(ID, messageDate, message, sender, receiver, sdf);
    }

    @Override
    public String toString() {
        return "Message{" +
                "ID=" + ID +
                ", messageDate=" + messageDate +
                ", message='" + message + '\'' +
                ", sender=" + sender +
                ", receiver=" + receiver +
                '}';
    }
}
