using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Data.SqlClient;

namespace FaReNEW
{
    public partial class WebForm5 : System.Web.UI.Page
    {
        SqlConnection conn = new SqlConnection(@"Data Source=LAPTOP-4LQIPA5M;Initial Catalog=face_recog;user id=sa;password=avi@123");
        SqlCommand cmd = new SqlCommand();
        protected void Page_Load(object sender, EventArgs e)
        {

        }

        protected void Button1_Click(object sender, EventArgs e)
        {
            Response.Redirect("Home.aspx");
        }

        protected void TextBox1_TextChanged(object sender, EventArgs e)
        {

        }

        protected void TextBox2_TextChanged(object sender, EventArgs e)
        {

        }

        protected void Button2_Click(object sender, EventArgs e)
        {
            cmd.Connection = conn;
            cmd.CommandText = "select *from Admin where User_Name='" + TextBox1.Text + "' and Password='" + TextBox2.Text + "'";

            conn.Open();
            SqlDataReader dr = cmd.ExecuteReader();
            if (dr.Read())
            {
                if (dr.HasRows == true)
                {
                    Session["user"] = dr[0].ToString();
                    Session["ps"] = dr[1].ToString();
                    Response.Redirect("Adminsucc.aspx");
                }
            }
            else
            {
                erlb.Text = "user name or password invalid....";
                TextBox1.Focus();


            }

            conn.Close();
        
    }

        protected void LinkButton5_Click(object sender, EventArgs e)
        {

        }

        protected void LinkButton4_Click(object sender, EventArgs e)
        {

        }

        protected void LinkButton3_Click(object sender, EventArgs e)
        {
            Response.Redirect("Discover.html");
        }

        protected void LinkButton2_Click(object sender, EventArgs e)
        {
            Response.Redirect("About_us.html");
        }

        protected void LinkButton1_Click(object sender, EventArgs e)
        {
            string email = "facecappers@gmail.com";
            ClientScript.RegisterStartupScript(this.GetType(), "mailto", "parent.location='mailto:" + email + "'", true);
        }
    }
}