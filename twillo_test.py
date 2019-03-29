#sending text messages using program
from twilio.rest import TwilioRestClient

account_sid = "" #replace with code from twilio acount sid
auth_token = "" #replace with code from twilio account auth code

client = TwilioRestClient(account_sid,auth_token)

message = client.sms.messages.create(body="body of message",to="+1tophonenumber",from_="+1twiliophonenumber")

print(message.sid)

