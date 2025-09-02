using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Data;
using System.Data.SqlClient;
using System.Configuration;

namespace FaReNEW
{
    public partial class WebForm3 : System.Web.UI.Page
    {
        SqlConnection conn = new SqlConnection();
        protected void Page_Load(object sender, EventArgs e)
        {
            conn.ConnectionString = (@"Data Source=LAPTOP-4LQIPA5M;Initial Catalog=face_recog;user id=sa;password=avi@123");
            conn.Open();
        }

        protected void TextBox1_TextChanged(object sender, EventArgs e)
        {

        }

        protected void TextBox2_TextChanged(object sender, EventArgs e)
        {

        }

        protected void TextBox3_TextChanged(object sender, EventArgs e)
        {

        }

        protected void TextBox4_TextChanged(object sender, EventArgs e)
        {

        }

        protected void TextBox5_TextChanged(object sender, EventArgs e)
        {

        }

        protected void TextBox6_TextChanged(object sender, EventArgs e)
        {

        }
        protected void TextBox5_TextChanged1(object sender, EventArgs e)
        {

        }

        protected void Button1_Click1(object sender, EventArgs e)
        {
            conn.Close();
            conn.Open();
            SqlCommand cd = new SqlCommand("insert into Registration values(@User_Name,@Password,@Email,@Mob,@DOB)", conn);
            cd.Parameters.AddWithValue("@User_Name", TextBox1.Text);
            cd.Parameters.AddWithValue("@Password", TextBox2.Text);
            cd.Parameters.AddWithValue("@Email", TextBox4.Text);
            cd.Parameters.AddWithValue("@Mob", TextBox5.Text);
            cd.Parameters.AddWithValue("@Dob", TextBox6.Text);
            cd.ExecuteNonQuery();
            conn.Close();
            Session["UN"] = TextBox1.Text.ToString();
            Session["PS"] = TextBox2.Text.ToString();
            Session["EM"] = TextBox4.Text.ToString();
            Session["MOB"] = TextBox5.Text.ToString();
            Session["DOB"] = TextBox6.Text.ToString();
            Response.Redirect("Regissucc.aspx");

        }

        protected void LinkButton2_Click(object sender, EventArgs e)
        {
            Response.Redirect("About_us.html");
        }

        protected void LinkButton3_Click(object sender, EventArgs e)
        {
            Response.Redirect("Discover.html");
        }

        protected void LinkButton1_Click(object sender, EventArgs e)
        {
            string email = "facecappers@gmail.com";
            ClientScript.RegisterStartupScript(this.GetType(), "mailto", "parent.location='mailto:" + email + "'", true);
        }
    }
}