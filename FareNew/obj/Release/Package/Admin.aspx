<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Admin.aspx.cs" Inherits="FaReNEW.WebForm5" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
    <link rel="stylesheet" href="Admin.css" />


</head>
<body>
    <form id="form1" runat="server">
        <div class="nav">
            <asp:Label ID="lb1" runat="server" CssClass="auto-style4" Text="FACE CAPPERS"></asp:Label>
            <asp:LinkButton ID="LinkButton1" runat="server" OnClick="LinkButton1_Click"  >Report An Error</asp:LinkButton>
            <asp:LinkButton ID="LinkButton2" runat="server" CssClass="auto-style5" OnClick="LinkButton2_Click" >About Us</asp:LinkButton>
            <asp:LinkButton ID="LinkButton3" runat="server" OnClick="LinkButton3_Click" CssClass="auto-style6" >Discover</asp:LinkButton>
            <asp:Button ID="Button1" runat="server" Text="Home" CssClass="auto-style4" OnClick="Button1_Click"  />
            </div>
        <div class="log">
            

            <asp:Label ID="Label1" runat="server" Text="Admin-Login" ></asp:Label>
            <asp:Label ID="Label2" runat="server" Text="Username" ></asp:Label>
            <asp:Label ID="Label3" runat="server" Text="Password"></asp:Label>
            <asp:TextBox ID="TextBox1" runat="server" Height="32px" Width="303px" OnTextChanged="TextBox1_TextChanged" ></asp:TextBox>
            <asp:TextBox ID="TextBox2" runat="server" TextMode="Password" OnTextChanged="TextBox2_TextChanged" ></asp:TextBox>
            <br />
            <asp:CheckBox ID="CheckBox1" runat="server" Text="Remember Me" ForeColor="LightCyan" />
            <br />
            <asp:Button ID="Button2" runat="server"  Text="Login" OnClick="Button2_Click" />
            
            
            <asp:Label ID="erlb" runat="server" ></asp:Label>
        </div>
    </form>
</body>
</html>
