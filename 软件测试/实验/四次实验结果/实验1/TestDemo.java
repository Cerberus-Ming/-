import org.junit.Assert;
import org.junit.Test;

public class TestDemo {

    @Test
    public void test() {
        // 测试用例
        Date date1 = new Date(2, 29, 2020); // 日期1
        Date date2 = new Date(2, 28, 2022); // 日期2（用于和日期1比较）
        Date date3 = new Date(12, 31, 2022); // 日期3
        Date date4 = new Date(3, 1, 2022);  // 日期4（用于和日期2的下一天比较）
        Date date5 = new Date(12, 31, -1);  // 日期5

        // 计算日期的下一天
        Date nd1 = Nextday.nextDay(date1);
        Date nd2 = Nextday.nextDay(date2);
        Date nd3 = Nextday.nextDay(date3);
        Date nd4 = Nextday.nextDay(date4);
        Date nd5 = Nextday.nextDay(date5);

        // 转换日期对象为字符串
        String result1 = nd1.toString();
        String result2 = nd2.toString();
        String result3 = nd3.toString();
        String result4 = nd4.toString();
        String result5 = nd5.toString();

        // 测试Date类打印日期方法
        nd1.printDate();
        nd2.printDate();
        nd3.printDate();
        nd4.printDate();
        nd5.printDate();

        // 使用断言来验证预期的结果
        // 测试Date类的toString方法
        Assert.assertEquals("3/1/2020", result1);
        Assert.assertEquals("3/1/2022", result2);
        Assert.assertEquals("1/1/2023", result3);
        Assert.assertEquals("3/2/2022", result4);
        Assert.assertEquals("1/1/1", result5);

        // 测试Date类的nextday方法
        Assert.assertTrue(nd2.equals(date4));
        // 测试Date类的equals方法
        Assert.assertFalse(date1.equals(date2));
        // 测试getDay、getYear、getMonth
        Assert.assertFalse(date1.getYear().equals(1));
        Assert.assertFalse(date1.getMonth().equals(1));
        Assert.assertFalse(date1.getDay().equals(1));

        // 测试异常处理
        // 验证不合法日期参数会抛出异常，并且异常包含正确的错误消息

        // 日期不规范（下溢）
        try {
            Date date = new Date(12, -1, 2020);
        } catch (IllegalArgumentException e) {
            Assert.assertEquals(e.getMessage(), "Not a valid day");
        }
        // 日期不规范（上溢）
        try {
            Date date = new Date(12, 32, 2020);
        } catch (IllegalArgumentException e) {
            Assert.assertEquals(e.getMessage(), "Not a valid day");
        }
        // 月份不规范（上溢）
        try {
            Date date = new Date(13, 1, 2020);
        } catch (IllegalArgumentException e) {
            Assert.assertEquals(e.getMessage(), "Not a valid month");
        }
        // 月份不规范（下溢）
        try {
            Date date = new Date(-1, 1, 2020);
        } catch (IllegalArgumentException e) {
            Assert.assertEquals(e.getMessage(), "Not a valid month");
        }
        // 年份不规范（下溢）
        try {
            Date date = new Date(1, 1, 0);
        } catch (IllegalArgumentException e) {
            Assert.assertEquals(e.getMessage(), "Not a valid year");
        }
    }
}
