<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="forgot_pass.aspx.cs" Inherits="FaReNEW.forgot_pass" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
    <link rel="stylesheet" href="forgot_pass.css"
</head>

<body>
    <form id="form1" runat="server">
        <div class="nav">

            <asp:Label ID="Label1" runat="server" Text="Face Cappers" ></asp:Label>
            <asp:LinkButton ID="LinkButton1"  runat="server" OnClick="LinkButton1_Click" CssClass="auto-style2" >Report An Error</asp:LinkButton>
            <asp:LinkButton ID="LinkButton2"  runat="server" OnClick="LinkButton2_Click">About Us</asp:LinkButton>
            <asp:LinkButton ID="LinkButton3"  runat="server" OnClick="LinkButton3_Click" CssClass="auto-style2" >Discover</asp:LinkButton>
        </div>
        <div class="reg">
            <asp:Label ID="Label2" runat="server"  Text="Forgot-Password"></asp:Label>
            <p id="para1">Enter your username and if your information <br />matches you will be able to change passwords</p>
            <hr />


            <asp:Label ID="Label3" runat="server" Text="User Name"></asp:Label>
            <asp:TextBox ID="TextBox1" runat="server" CssClass="auto-style2"  ></asp:TextBox>
            <asp:Label ID="Label4" runat="server" Text="Password"></asp:Label>
            <asp:TextBox ID="TextBox2" runat="server" TextMode="Password" CssClass="auto-style1" Visible="False" ></asp:TextBox>
            <asp:Label ID="Label5" runat="server" Text="Confirm Password"></asp:Label>
            <asp:TextBox ID="TextBox3" runat="server"  TextMode="Password" Visible="False" ></asp:TextBox>
            <asp:Label ID="Label6" runat="server" Text="Email"></asp:Label>
            <asp:TextBox ID="TextBox4" runat="server" CssClass="auto-style1" ></asp:TextBox>
            <asp:Label ID="Label7" runat="server" Text="Mobile Number"></asp:Label>
            <asp:TextBox ID="TextBox5" runat="server"  OnTextChanged="TextBox5_TextChanged1"   ></asp:TextBox>
            <asp:Label ID="Label8" runat="server" Text="Date of Birth"></asp:Label>
            <asp:TextBox ID="TextBox6" runat="server"  TextMode="Date"  OnTextChanged="TextBox6_TextChanged"  ></asp:TextBox>
            <asp:Button ID="Button1" runat="server" Text="Check" OnClick="Button1_Click1" CssClass="auto-style1" />
            <asp:Button ID="Button2" runat="server" Text="Register" OnClick="Button2_Click" />
            <asp:RequiredFieldValidator ID="RequiredFieldValidator2" runat="server" ControlToValidate="TextBox2" ErrorMessage="Please enter password" ForeColor="Red"></asp:RequiredFieldValidator>
            <asp:CompareValidator ID="CompareValidator1" runat="server" ControlToCompare="TextBox2" ControlToValidate="TextBox3"  ErrorMessage="CompareValidator" ForeColor="Red">Password not matching</asp:CompareValidator>
            <asp:RequiredFieldValidator ID="RequiredFieldValidator3" runat="server" ControlToValidate="TextBox6" CssClass="auto-style1" ErrorMessage="Enter Date of Birth" ForeColor="Red"></asp:RequiredFieldValidator>
            <asp:RequiredFieldValidator ID="RequiredFieldValidator1" runat="server" ControlToValidate="TextBox1" ErrorMessage="Enter your username" ForeColor="Red"></asp:RequiredFieldValidator>
        </div>

    </form>
</body>
</html>
