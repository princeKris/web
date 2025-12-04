from django.db import models
from django.contrib.auth.models import AbstractUser
from pdf2image import convert_from_path
from PIL import Image
import os
import fitz 
from django.conf import settings

from django.utils import timezone

def pdf_upload_path(instance, filename):
    return f"pdfs/{filename}"

def preview_upload_path(instance, filename):
    return f"pdf_previews/{filename}"

def userimg_upload_path(instance, filename):
    return f"userimage/{filename}"

class User(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('staff', 'Staff'),
        ('user', 'User'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    image = models.ImageField(upload_to=userimg_upload_path, blank=True, null=True)
    bio = models.CharField(max_length=255, blank=True, null=True)
    bdo=models.DateTimeField(blank=True, null=True)
    date_joined=models.DateTimeField(auto_now_add=True)
    verified = models.BooleanField(default=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    blocked = models.BooleanField(default=False)
    blocked_at = models.DateTimeField(null=True, blank=True)
    deleted= models.BooleanField(default=False)
    def __str__(self):
        return f"{self.username}'s profile"
    
    def save(self, *args, **kwargs):
        if self.role=='admin' or self.role=='staff':
            self.verified=True
        if self.verified and self.verified_at is None:
            self.verified_at = timezone.now()
        if self.blocked and self.blocked_at is None:
            self.blocked_at = timezone.now()
        super().save(*args, **kwargs)






class Post(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to=pdf_upload_path)
    image = models.ImageField(upload_to=preview_upload_path, blank=True, null=True)   # ADD THIS
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="posts")
    likes = models.ManyToManyField(User, related_name="liked_posts", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.file and not self.image:
            pdf_path = self.file.path

            try:
                doc = fitz.open(pdf_path) # open PDF
                page = doc.load_page(0) # first page
                pix = page.get_pixmap() # convert to image

                # prepare filename
                img_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".jpg"
                img_path = os.path.join(settings.MEDIA_ROOT, "pdf_previews", img_filename)

                # ensure directory exists
                os.makedirs(os.path.dirname(img_path), exist_ok=True)

                pix.save(img_path)  # save jpeg

                self.image = f"pdf_previews/{img_filename}"
                super().save(update_fields=["image"])

            except Exception as e:
                print("Preview Error:", e)


class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name="comments")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    parent = models.ForeignKey('self', null=True, blank=True,related_name="replies", on_delete=models.CASCADE)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=True)
