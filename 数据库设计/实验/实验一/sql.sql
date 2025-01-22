1.1
select sNo,sName
from student
where age>=20;

1.2 
select sNo,sName,2022-age
from student
where sName Like'钱%'
and sex='男';

1.3
select cNo,cName
from course
where credit>3;

1.4
select sNo,sName
from student
where dNo is null;

1.5
select dNo,dName
from department
where homepage is null;

2.1
select dNo,avg(age)
from student
where dNo is not null
group by dNo;

2.2
select sNo,avg(score)
from sc
group by sNo;

2.3
select cNo,avg(score)
from sc
group by cNo;

2.4
select dNo,count(cNo)
from course
group by dNo
order by dNo;

2.5
select cNo,count(sNo)
from sc
group by cNo
order by cNo;

3.1
select sNo,sName
from student
where dNo in
         (select dNo
         from department
         where dName='信息学院');
3.2
select cNo,cName
from course
where dNo in
         (select dNo
         from department
         where dName='软件学院');

3.3
select sNo,sName
from student
where dNo in
         (select dNo
         from student
         where sName='陈丽');

3.4
select sNo,sName
from student
where age in
          (select age 
          from student
          where sName='张三丰');

3.5
select sNo,sName
from student
where age in
          (select age 
          from student
          where sName='张三丰')
and dNo not in
           (select dNo
           from student
           where sName='张三丰');

3.7
select cNo,cName
from course
where credit >
            (select credit
            from course
            where cName='离散数学');

3.7
select count(sNo)
from sc
where cNo in
         (select cNo
         from course
         where cName='离散数学');

3.8
select sNo,sName
from student
where sNo not in
             (select sNo
             from sc
             where cNo in
                       (select cNo
                       from course
                       where cName='离散数学'));

3.9
select cNo,cName,credit
from course
where credit <>
             (select credit
             from course
             where cName='算法设计与分析')
and credit <>
           (select credit
           from course
           where cName='移动计算');

3.10
select cNo,cName
from course
where cNo in
                 (select cNo
                  from sc
                  group by cNo
                  having avg(score) >=90);

3.11
select s.sNo,s.sName,sc.score
from student s, sc
where s.sNo = sc.sNo
and sc.cNo =(select cNo
             from course
             where cName='离散数学');

3.12
select c.cNo,c.cName ,sc.score
from sc,course c
where sc.cNo=c.cNo
and sNo in (select sNo
           from student
           where sName='王兵');

3.13
select s.sNo,s.sName,c.cNo,c.cName,sc.score,s.sNo
from student s,course c,sc
where s.sNo=sc.sNo
and sc.score<60
and sc.cNo=c.cNo;

3.14
select distinct s.sName,s.sNo
from student s,department d,course c,sc
where d.dName='文学院'
and c.dNo=d.dNo
and sc.cNo=c.cNo
and s.sNo=sc.sNo;

3.15
select s.sNo,s.sName,c.cNo,c.cName
from student s,sc,course c
where c.cNo,No=sc.cNo
and s.sNo=sc.sNo
and s.dNo in
         (select dNo
         from department
         where dName='信息学院')
and c.cNo in
         (select cNo
         from course
         where dNo in
                  (select dNo
                  from department
                  where dName='信息学院'));   

4.1
select s.*,sc.*
from student s left join sc
on s.sNo = sc.sNo;

4.2
select cNo,cName
from course
where cNo in
         (select cpNo
         from course
         where cName='形式语言与自动机');

4.3
select cNo,cName
from course
where cNo in
          (select cpNo
          from course
          where cNo in
                    (select cpNo
                    from course
                    where cName='形式语言与自动机'));

4.4
select cNo,cName
from course
where cpNo in
          (select cNo
          from course
          where cName='编译原理');

4.5
select cNo,cName
from course
where cpNo in
          (select cNo
          from course
          where cpNo in
                    (select cNo
                    from course
                    where cName='离散数学'));

4.6
select cNo,cName
from course
where cpNo is null;

4.7
select sNo,sName
from student
where sNo not in
             (select sNo
             from sc
             where cNo in
                       (select cNo
                       from course
                       where cName='形式语言与自动机'));
4.8
select s.sno,s.sname
from sc join student s
on not exists
            (select *
            from sc,course c1
            where sc.cno=c1.cno
            and sc.sno=s.sno
            and c1.cname!='离散数学')
and exists
         (select *
         from sc,course c2
         where sc.cno =c2.cno
         and sc.sno = s.sno
         and c2.cname='离散数学')
where s.sno=sc.sno;

4.9
select sNo,sName
from student
where sNo in
             (select sNo
             from sc
             where cNo in
                       (select cNo
                       from course
                       where cName='形式语言与自动机'))
and sno not in
            (select sno
            from sc
            where cno in
                     (select cpno
                     from course
                     where cno in
                                (select cno
                                from course
                                where cname='形式语言与自动机')));

4.10
select s.sNo,s.sName,sum(credit)
from student s,course c,sc
where s.sNo=sc.sNo
and sc.cNo=c.cNo
group by s.sNo
having sum(credit)>=28;

4.11
select sno,sname
from student
where sno in
         (select sno
         from sc
         where score>=85
         group by sno
         having count(cno)>3);

4.12
select sno,sname
from student
where sno in
         (select sno
         from sc
         where score>=60
         group by sno
         having count(cno)=3);

4.13
select d.dno,d.dname,count(sno)
from department d,student s
where s.dno=d.dno
group by d.dno
having count(sno)>6;

4.14
select s.sNo,s.sName
from student s,sc
where s.sNo = sc.sNo
group by s.sno
having avg(score)>all(
		     select avg(score)
		     from student join sc
		     on student.sno=sc.sno
		     where sname='王兵');

4.15
select sno,sname
from student
where sno in
          (select sno
          from sc
          where cno in
                    (select cno
                    from course
                    where cname='离散数学'))
and sno in
        (select sno
        from sc
        where cno in
                  (select cno
                  from course
                  where cname='编译原理'));
		    
4.16
select avg(score)
from sc
where cno in
          (select cno
          from course
          where cname='离散数学')
and sno in
        (select sno
        from student
        where dno in
                  (select dno
                  from department
                  where dname='软件学院'));

4.17
select s.sno,s.sname,s.age,d.dno,d.dname
from student s left join department d
on s.dno =d.dno
where age not in
             (select age
             from student
             where dno in
                      (select dno
                      from department
                      where dname='软件学院'));

4.18
select d.dno,d.dname,c.cno,c.cname,count(s.sno)
from department d join student s
on d.dno = s.dno join sc
on s.sno = sc.sno join course c
on sc.cno = c.cno
group by d.dno,c.cno
having count(s.sno)>4;

4.19
select s.sNo,s.sName,d.dNo,d.dName
from sc join student s 
on s.sno=sc.sno 
left join department d 
on s.dNo = d.dNo
where sc.sNo not in
                (select sNo
                from sc
                where cNo not in (select cNo
                               from course
                               where cName='高等数学'));

4.20
select s.sno,s.sname,sum(sc.score*c.credit)/sum(c.credit)
from student s,sc,course c
where sc.cno=c.cno
and score is not null
and credit is not null
and sc.sno=s.sno
group by s.sno
having (sum(sc.score*c.credit)/sum(c.credit))<70;

4.21
select s.sno,s.sname
from student s
where not exists
               (select *
               from course c
               where c.dno in
                           (select dno
                           from department
                           where dname='信息学院')
              and not exists
                           (select *
                           from sc
                           where sc.sno =s.sno
                           and sc.cno=c.cno));

4.22
select s.sno,s.sname
from student s,sc sc1
where not exists
                (select *
                from student s,sc sc2
                where s.sname='杨佳伟'
                and not exists
                             (select *
                             from sc sc3
                             where sc3.sno=sc1.sno
                             and sc3.cno=sc2.cno));

DDL练习
1.
CREATE TABLE publishingHouse(
pNo CHAR(6) NOT NULL UNIQUE,
pName VARCHAR(20) NOT NULL,
pAddress VARCHAR(20) NOT NULL,
pTelephone CHAR(20),
PRIMARY KEY(pNo)
);

CREATE TABLE book(
bNo CHAR(6) NOT NULL UNIQUE,
bName VARCHAR(20) NOT NULL,
bAuthor VARCHAR(20) NOT NULL,
bISBN CHAR(20) NOT NULL UNIQUE,
pNo CHAR(6) NOT NULL,
version CHAR(3) NOT NULL,
PRIMARY KEY(bNo),
FOREIGN KEY(pNo) REFERENCES publishingHouse(pNo)
);

select * from book,publishingHouse;

DROP TABLE publishingHouse CASCADE;

DROP TABLE book CASCADE;

select * from book,publishingHouse;

CREATE TABLE publishingHouse(
pNo CHAR(6) NOT NULL UNIQUE,
pName VARCHAR(20) NOT NULL,
pAddress CHAR(20) NOT NULL,
pTelephone CHAR(11) NOT NULL
);

CREATE TABLE book(
bNo CHAR(6) NOT NULL UNIQUE,
bName VARCHAR(20) NOT NULL,
bAuthor VARCHAR(20) NOT NULL,
bISBN CHAR(20) NOT NULL UNIQUE,
pNo CHAR(6) NOT NULL,
version CHAR(3) NOT NULL,
bDate date NOT NULL
);

ALTER TABLE publishingHouse ADD PRIMARY KEY(pNo);

ALTER TABLE book ADD PRIMARY KEY(bNO);

ALTER TABLE book ADD FOREIGN KEY(pNo) REFERENCES publishingHouse(pNo);

select * from publishingHouse,book;

2.
INSERT INTO publishingHouse VALUES('01','西工大出版社','西安','123456');

INSERT INTO publishingHouse VALUES('02','复旦出版社','上海','456789');

INSERT INTO book VALUES('001','数据库系统基础','cj','1245687','01','1','2022-4-1');

INSERT INTO book VALUES('002','婚姻与家庭','rsx','7346746','02','1','2022-4-1');

select * from publishingHouse;

select * from book;

UPDATE publishingHouse
SET pAddress='苏州'
where pNo='01';

select * from publishingHouse;

DELETE FROM book;

DELETE FROM publishingHouse;

select * from publishingHouse;

select * from book; 

3.
CREATE VIEW s_select as
select *
from student
where sno in
         (select s.sno
         from student s,department d,course c,sc
         where d.dName='软件学院'
         and d.dNo=s.dNo
         and c.cName='离散数学'
         and c.cNo=sc.cNo
         and sc.sNo=s.sNo);
         
INSERT INTO s_select VALUES ('200201','陈靖','男','20','02');

select * from s_select;

CREATE VIEW avg_score as
select c.*,avg(score)
from sc natural join course c
group by c.cno;

select * from avg_score;

4.
CREATE TABLE s_score(
sNo CHAR(6),
sName CHAR(20),
avgscore INT);

INSERT INTO s_score(sNo,sName,avgscore)
select s.sno,s.sname,avg(score)
from sc natural join student s
group by s.sno;

select * from s_score;

DCL
Begin Transaction;
select * from student;
select * from course;
commit;

Begin Transaction;
select * from student;
select * from course;
rollback;