import base64



css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
def get_bot_template(MSG,img_base64):
    bot_template = f'''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="{img_base64}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
        <div class="message">{MSG}</div>
    </div>
    '''
    return bot_template

def get_user_template(MSG,img_base64):
    user_template = f'''
    <div class="chat-message user">
        <div class="avatar">
            <img src="{img_base64}">
        </div>    
        <div class="message">{MSG}</div>
    </div>
    '''
    return user_template

# https://drive.google.com/uc?id=1B6DObmgGFlYUH3_qADs60StUYNbakMCA

# https://drive.google.com/file/d/1B6DObmgGFlYUH3_qADs60StUYNbakMCA/view?usp=drive_link

# <img src="https://drive.google.com/file/d/1J7CQQl78JfOjwy7hKLcF1xHGUYlIQVXH/view?usp=drive_link">


