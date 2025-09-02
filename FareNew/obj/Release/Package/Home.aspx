<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Home.aspx.cs" Inherits="FaReNEW.WebForm1" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
    <link rel="stylesheet" href="Home.css" />
</head>
<body>
    <form id="form1" runat="server">
        <div class="nav">
            <asp:Label ID="lb1" runat="server"  Text="FACE CAPPERS" ></asp:Label>
            <asp:LinkButton ID="LinkButton1" runat="server"  OnClick="LinkButton1_Click" >Report An Error</asp:LinkButton>
            <asp:LinkButton ID="LinkButton2" runat="server"  OnClick="LinkButton2_Click">About Us</asp:LinkButton>
            <asp:LinkButton ID="LinkButton3" runat="server"  OnClick="LinkButton3_Click" >Discover</asp:LinkButton>
            <asp:Button ID="Button1" runat="server" CssClass="auto-style5" OnClick="Button1_Click" Text="Admin-Login"  />
        </div>
        <div class="log">
            

            <asp:Label ID="Label1" runat="server" >Login</asp:Label>
            <asp:Label ID="Label2" runat="server" Text="Username" CssClass="auto-style12"></asp:Label>
            <asp:Label ID="Label3" runat="server" Text="Password"></asp:Label>
            <asp:TextBox ID="TextBox1" runat="server"  OnTextChanged="TextBox1_TextChanged1"></asp:TextBox>
            <asp:TextBox ID="TextBox2" runat="server" TextMode="Password" OnTextChanged="TextBox2_TextChanged1"></asp:TextBox>
            <br />
            <asp:CheckBox ID="CheckBox1" runat="server" Text="Remember Me" ForeColor="LightCyan" OnCheckedChanged="CheckBox1_CheckedChanged1" />
            <asp:LinkButton ID="LinkButton4" runat="server" OnClick="LinkButton4_Click2">Forgot Password?</asp:LinkButton>
            <br />
            <asp:Button ID="Button2" runat="server"  Text="Face-Identify"  OnClick="Button2_Click1"  />
            
            
            <asp:Label ID="erlb" runat="server" ></asp:Label>
            

            <asp:Label ID="Label4" runat="server" Text="New to Face Cappers?"></asp:Label>
            <asp:LinkButton ID="LinkButton5" runat="server" OnClick="LinkButton5_Click1" >Sign-up</asp:LinkButton>
            

            
            

            <asp:Button ID="Bt3" runat="server" OnClick="Bt3_Click" Text="Face-Recognition" />
            

            
            

        </div>
    </form>
</body>
</html>
