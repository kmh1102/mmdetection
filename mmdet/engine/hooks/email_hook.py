# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@HOOKS.register_module()
class EmailNotificationHook(Hook):
    def __init__(self, recipient_email, send_address, password):
        self.recipient_email = recipient_email
        self.send_address = send_address
        self.password = password

    def send_email(self, subject, body):
        try:
            message = MIMEMultipart()
            message['From'] = self.send_address
            message['To'] = self.recipient_email
            message['Subject'] = subject

            message.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP_SSL('smtp.qq.com', 465)
            server.login(self.send_address, self.password)
            server.sendmail(self.send_address, self.recipient_email, message.as_string())
            print("Email sent successfully!")
        except smtplib.SMTPException as e:
            print(f"Error: {e}")
        finally:
            server.quit()

    def after_run(self, runner):
        subject = "MMDetection Training Completed"
        body = f"Training task has completed."
        self.send_email(subject, body)

