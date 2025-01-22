import java.io.Serializable;
import java.util.Objects;

/**
 * @description: 用户类
 * @author Eternity
 * @date 2023/11/24 14:24
 * @version 1.0
 */
public class User implements Serializable {

    private String username; // 用户名

    private String password; // 密码

    /**
     * 构造函数
     * @param username
     * @param password
     */
    public User(String username, String password) {
        this.username = username;
        this.password = password;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return Objects.equals(username, user.username) && Objects.equals(password, user.password);
    }

    /**
     * 重写 hashCode 方法，用于获取用户对象的哈希码
     * @return
     */
    @Override
    public int hashCode() {
        return Objects.hash(username, password);
    }

    @Override
    public String toString() {
        return "User{" +
                "username='" + username + '\'' +
                ", password='" + password + '\'' +
                '}';
    }
}

