# app/permissions.py

from rest_framework.permissions import BasePermission, SAFE_METHODS
from django.contrib.auth import get_user_model
from .models import Post, Comment   

User = get_user_model()


class UnifiedRBACPermission(BasePermission):

    def has_permission(self, request, view):
        # Allow public access to GET, HEAD, OPTIONS
        if request.method in SAFE_METHODS:
            return True

        # Block unsafe methods (POST, PUT, PATCH, DELETE) if not logged in
        return request.user.is_authenticated


    def has_object_permission(self, request, view, obj):
        user = request.user

        # POST RBAC
        if isinstance(obj, Post):

            # EDIT post Only owner
            if request.method in ["PUT", "PATCH"]:
                return obj.author == user

            # DELETE post
            if request.method == "DELETE":

                # Admin can delete all posts
                if user.role == "admin":
                    return True

                # Staff delete rules:
                if user.role == "staff":
                    # Staff cannot delete admin's posts
                    if obj.author.role == "admin":
                        return False

                    # Staff can delete own post OR normal users' posts
                    return obj.author == user or obj.author.role == "user" #True

                # User can delete only their own post
                return obj.author == user

            return True

        # COMMENT RBAC
        if isinstance(obj, Comment):
            comment_author = obj.user
            post_author = obj.post.author

            if user.role == "admin":
                return True

            if user.role == "staff":
                if comment_author.role == "admin":
                    return False
                return True

            if user == comment_author:
                return True

            if user == post_author:
                return True

            return False

        # USER DELETE RBAC
        if isinstance(obj, User):

            if request.method == "DELETE" and user.role == "admin":
                return True

            return False

        return False
