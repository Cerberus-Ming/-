--DROP TABLE PUBLISHER CASCADE;
--DROP TABLE BOOK CASCADE;
--DROP TABLE AUTHOR CASCADE;
--DROP TABLE CATEGORY CASCADE;
--DROP TABLE CUSTOMER CASCADE;
--DROP TABLE PROMOTION CASCADE;
--DROP TABLE ORDERS CASCADE;
--DROP TABLE SHIPMENT CASCADE;
--DROP TABLE LINEITEM CASCADE;
--DROP TABLE PROMOTE CASCADE;
--DROP TABLE WRITES CASCADE;
--DROP TABLE BELONGTO CASCADE;

Create Table Publisher(
Publisher_id CHAR(10) NOT NULL,
PublisherName VARCHAR(20) NOT NULL,
Address VARCHAR(30) NOT NULL,
Primary Key(Publisher_id)
);

Create Table Book(

Title VARCHAR(20) NOT NULL,
ISBN VARCHAR(20) NOT NULL,
Edition INT,
PublishDate DATE,
Price decimal(10,3),
BookDescription VARCHAR(80),
Publisher_id CHAR(10),
Primary Key(ISBN),
Foreign Key(Publisher_id) References Publisher(Publisher_id)
);

Create Table Author(
AuthorName VARCHAR(30) NOT NULL,
Author_id CHAR(10) NOT NULL,
Primary Key(Author_id)
);

Create Table Customer(
Customer_id CHAR(10) NOT NULL,
CustomName VARCHAR(30) NOT NULL,
MailingAddress VARCHAR(30),
CreditCardNumber VARCHAR(20) NOT NULL UNIQUE,
PhoneNumber CHAR(11),
EmailAddress VARCHAR(20),
Primary Key(Customer_id)
);

Create Table Orders(
OrderNumber CHAR(20) NOT NULL,
MailingAddress VARCHAR(40),
Method VARCHAR(15),
DateAndTimeOrder Date NOT NULL,
Customer_id CHAR(10),
Primary Key(OrderNumber),
Foreign Key(Customer_id) References Customer(Customer_id)
);

Create Table Shipment(
ShipmentDate Date NOT NULL,
TrackingNumber CHAR(20) NOT NULL,
ExpectedDate Date,
OrderNumber CHAR(10) NOT NULL,
Primary Key(TrackingNumber),
Foreign Key(OrderNumber) References Orders(OrderNumber)
);

Create Table Promotion(
Promotion_id CHAR(10) NOT NULL,
PercentageDiscount Float,
Primary Key(Promotion_id)
);

Create Table Category(
Category_id CHAR(10) NOT NULL,
CategoryName VARCHAR(20),
Super_id CHAR(10),
Primary Key(Category_id),
Foreign Key(Super_id) References Category(Category_id)
);

Create Table LineItem(
ISBN VARCHAR(16),
OrderNumber CHAR(20),
TrackingNumber CHAR(20),
TotalPrice Float,
Quantity Int,
Primary Key(ISBN,OrderNumber),
Foreign Key(ISBN) References Book(ISBN),
Foreign Key(OrderNumber) References Orders(OrderNumber),
Foreign Key(TrackingNumber) References Shipment(TrackingNumber)
);

Create Table Write(
ISBN VARCHAR(16),
Author_id CHAR(10),
Primary Key(ISBN,Author_id),
Foreign Key(ISBN) References Book(ISBN),
Foreign Key(Author_id) References Author(Author_id)
);

Create Table BelongTo(
ISBN VARCHAR(16),
Category_id CHAR(10),
Primary Key(ISBN,Category_id),
Foreign Key(ISBN) References Book(ISBN),
Foreign Key(Category_id) References Category(Category_id)
);

Create Table Promote(
ISBN VARCHAR(20),
Promotion_id CHAR(10),
Primary Key(ISBN,Promotion_id),
Foreign Key(ISBN) References Book(ISBN),
Foreign Key(Promotion_id) References Promotion(Promotion_id)
);

Create Table TotalBookPrice(
BookPrice decimal(10,3),
Quantity Int,
OrderNumber CHAR(20),
ISBN VARCHAR(20),
Primary Key(OrderNumber,ISBN),
Foreign Key(OrderNumber) References Orders(OrderNumber)
Foreign Key(ISBN) References Book(ISBN)
);

Create Table Duration(
StartDate Date NOT NULL,
EndDate Date NOT NULL,
Promotion_id CHAR(10),
Primary Key(StartDate,EndDate,Promotion_id),
Foreign Key(Promotion_id) References Promotion(Promotion_id)
);
