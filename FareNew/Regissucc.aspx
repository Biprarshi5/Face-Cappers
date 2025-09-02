<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Regissucc.aspx.cs" Inherits="FaReNEW.WebForm4" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <link rel="stylesheet" href="Regissucc.css" />
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
         <div class="auto-style1">
            <asp:Label ID="lb1" runat="server"  Text="FACE CAPPERS"></asp:Label>
            <asp:LinkButton ID="LinkButton1" runat="server" OnClick="LinkButton1_Click"   >Report An Error</asp:LinkButton>
            <asp:LinkButton ID="LinkButton2" runat="server" OnClick="LinkButton2_Click"  >About Us</asp:LinkButton>
            <asp:LinkButton ID="LinkButton3" runat="server" OnClick="LinkButton3_Click"  >Discover</asp:LinkButton>   
             <asp:Button ID="Button1" runat="server" OnClick="Button1_Click" Text="HOME" CssClass="auto-style1" style="left: 0px; top: 0px" />
            
             <asp:Label ID="Label1" runat="server"  Text="User-Name :"></asp:Label>
             <asp:Label ID="Label8" runat="server"></asp:Label>
             <asp:Label ID="Label2" runat="server" ></asp:Label>
             <asp:Label ID="Label3" runat="server"  Text="Password :"></asp:Label>
             <asp:Label ID="Label4" runat="server"></asp:Label>
             <asp:Label ID="Label5" runat="server"  Text="E-Mail :"></asp:Label>
             <asp:Label ID="Label6" runat="server" ></asp:Label>
             <asp:Label ID="Label7" runat="server" Text="Mobile :"></asp:Label>
             <asp:Label ID="Label9" runat="server" Text="D-O-B :"></asp:Label>
             <asp:Label ID="Label10" runat="server"></asp:Label>
            
             <asp:Label ID="Label11" runat="server" Text="Press Home to go back to LOGIN PAGE"></asp:Label>
            
        </div>
    </form>
</body>
</html>
