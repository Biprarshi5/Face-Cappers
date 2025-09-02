<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Logsucc.aspx.cs" Inherits="FaReNEW.WebForm2" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
    <link rel="stylesheet" href="Logsucc.css" />
    <style type="text/css">
        .auto-style1 {
            left: 0px;
            top: 0px;
            height: 26px;
        }
    </style>
</head>
<body>
    <form id="form1" runat="server">
        <div class="nav">

            <asp:Label ID="Label1" runat="server" Text="Face Cappers"></asp:Label>
            <asp:LinkButton ID="LinkButton1" runat="server" OnClick="LinkButton1_Click" >Report An Error</asp:LinkButton>
            <asp:LinkButton ID="LinkButton2" runat="server"  OnClick="LinkButton2_Click">About Us</asp:LinkButton>
            <asp:LinkButton ID="LinkButton3" runat="server"  OnClick="LinkButton3_Click" CssClass="auto-style1" >Discover</asp:LinkButton>
            <asp:Button ID="Button1" runat="server" OnClick="Button1_Click" Text="Log-Out" />
        </div>
        <div class="container">
      <video id="video" height="500" width="500" autoplay muted></video>
    </div>
    <div class="result-container">
      <div id="pname">--Person Name--</div>
      <div id="emotion">--Detect Person Expression--</div>
      <div id="gender">--Detect Person Sex--</div>
      <div id="age">--Detect person Age--</div>
    </div>

    <script  src="./js/face-api.min.js/"></script>
    <script  src=".js/main.js/"></script>
        

        
    </form>
            
</body>
</html>
