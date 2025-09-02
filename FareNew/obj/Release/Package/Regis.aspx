<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Regis.aspx.cs" Inherits="FaReNEW.WebForm3" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <link rel="stylesheet" href="Regis.css" />
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        <div class="nav">
            <asp:Label ID="Label1" runat="server"  Text="Face Cappers" style="left: 0px; top: 0px; width: 221px"></asp:Label>
            <asp:LinkButton ID="LinkButton1" runat="server" OnClick="LinkButton1_Click" >Report An Error</asp:LinkButton>
            <asp:LinkButton ID="LinkButton2" runat="server" OnClick="LinkButton2_Click">About Us</asp:LinkButton>
            <asp:LinkButton ID="LinkButton3" runat="server" OnClick="LinkButton3_Click">Discover </asp:LinkButton>
        </div>
        <div class="reg">
            <asp:Label ID="Label2" runat="server" CssClass="auto-style3" Text="Register"></asp:Label>
            <p id="para1">Join Face Cappers and enjoy meeting new people <br />"TENSION FREE"</p>
            <hr />


            <asp:Label ID="Label3" runat="server" CssClass="auto-style4" Text="User Name"></asp:Label>
            <asp:TextBox ID="TextBox1" runat="server" CssClass="auto-style5" ></asp:TextBox>
            <asp:Label ID="Label4" runat="server" Text="Password"></asp:Label>
            <asp:TextBox ID="TextBox2" runat="server" TextMode="Password" ></asp:TextBox>
            <asp:Label ID="Label5" runat="server" Text="Confirm Password"></asp:Label>
            <asp:TextBox ID="TextBox3" runat="server" CssClass="auto-style7" style="left: 550px; top: 215px" TextMode="Password" ></asp:TextBox>
            <asp:Label ID="Label6" runat="server" Text="Email"></asp:Label>
            <asp:TextBox ID="TextBox4" runat="server" ></asp:TextBox>
            <asp:Label ID="Label7" runat="server" Text="Mobile Number"></asp:Label>
            <asp:TextBox ID="TextBox5" runat="server" CssClass="auto-style9" OnTextChanged="TextBox5_TextChanged1" ></asp:TextBox>
            <asp:Label ID="Label8" runat="server" Text="Date of Birth"></asp:Label>
            <asp:TextBox ID="TextBox6" runat="server" CssClass="auto-style8" TextMode="Date" style="left: 550px; top: 215px" ></asp:TextBox>
            <asp:Button ID="Button1" runat="server" Text="Register" OnClick="Button1_Click1" CssClass="auto-style5"  />

            <asp:RequiredFieldValidator ID="RequiredFieldValidator1" runat="server" ControlToValidate="TextBox1" ErrorMessage="please enter user name" ForeColor="#FF3300" CssClass="auto-style1"></asp:RequiredFieldValidator>
            <asp:RequiredFieldValidator ID="RequiredFieldValidator2" runat="server" ControlToValidate="TextBox2" ErrorMessage="please enter password" ForeColor="#FF3300"></asp:RequiredFieldValidator>
            <asp:CompareValidator ID="CompareValidator1" runat="server" ClientIDMode="Static" ControlToCompare="TextBox2" ControlToValidate="TextBox3" ErrorMessage="password not matching" ForeColor="#FF3300"></asp:CompareValidator>
            <asp:RegularExpressionValidator ID="RegularExpressionValidator1" runat="server" ControlToValidate="TextBox4" ErrorMessage="enter correct email" ForeColor="#FF3300" ValidationExpression="\w+([-+.']\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*"></asp:RegularExpressionValidator>
            <asp:RegularExpressionValidator ID="RegularExpressionValidator2" runat="server" ControlToValidate="TextBox5" ErrorMessage="enter valid mobile number" ForeColor="#FF3300" ValidationExpression="\d{10}"></asp:RegularExpressionValidator>
            <asp:RequiredFieldValidator ID="RequiredFieldValidator3" runat="server" ControlToValidate="TextBox6" ErrorMessage="enter birth date" ForeColor="#FF3300"></asp:RequiredFieldValidator>

        </div>
    </form>
</body>
</html>
