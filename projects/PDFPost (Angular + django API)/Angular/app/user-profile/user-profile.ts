import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { SharedImports } from '../shared/shared-imports';
import { ProfileService } from '../services/profile-service';
import { Auth } from '../services/auth';
import { formatDistanceToNow } from 'date-fns';

declare var bootstrap: any;

@Component({
  selector: 'app-user-profile',
  imports: [...SharedImports],
  templateUrl: './user-profile.html',
  styleUrls: ['./user-profile.scss'],
})
export class UserProfile implements OnInit {

  user: any = {};
  posts: any[] = [];

  // stats
  totalPosts = 0;
  totalLikes = 0;
  totalComments = 0;

  // Edit profile fields
  previewImage: string | null = null;
  selectedFile: File | null = null;

  editData = {
    username: '',
    email: '',
    dob: '',
    bio: '',
  };

  // Add post fields
  newPost = { title: '', description: '' };
  selectedPdf: File | null = null;
  selectedPdfName: string | null = null;

  // Edit post fields
  editPostData: any = {};
  editPdfFile: File | null = null;
  editPdfName: string = '';

  constructor(
    private profileApi: ProfileService,
    private auth: Auth,
  ) {}

  ngOnInit() {
    this.loadProfile();
    this.loadMyPosts();
  }

  formatForInput(dateString: string) {
    if (!dateString) return '';
    return dateString.split('T')[0];
  }

  // -------------------------------
  // LOAD PROFILE
  // -------------------------------
  loadProfile() {
    this.profileApi.myProfile().subscribe({
      next: (data: any) => {
        this.user = data;

        this.editData = {
          username: data.username,
          email: data.email,
          dob: this.formatForInput(data.bdo),
          bio: data.bio || '',
        };
      }
    });
  }

  // -------------------------------
  // LOAD POSTS
  // -------------------------------
  loadMyPosts() {
    this.profileApi.getMyPost().subscribe({
      next: (posts: any) => {
        this.posts = posts;

        this.totalPosts = posts.length;
        this.totalLikes = posts.reduce((sum:any, p:any) => sum + p.likes_count, 0);
        this.totalComments = posts.reduce((sum:any, p:any) => sum + p.comments.length, 0);
      }
    });
  }

  // -------------------------------
  // SELECT IMAGE PREVIEW
  // -------------------------------
  onImageSelect(event: any) {
    const file = event.target.files?.[0];
    if (!file) return;

    this.selectedFile = file;
    const reader = new FileReader();
    reader.onload = () => this.previewImage = reader.result as string;
    reader.readAsDataURL(file);
  }

  // -------------------------------
  // UPDATE PROFILE
  // -------------------------------
  updateProfile(e?: Event) {
    e?.preventDefault();

    const formData = new FormData();
    formData.append('username', this.editData.username);
    formData.append('email', this.editData.email);
    formData.append('bdo', this.editData.dob);
    formData.append('bio', this.editData.bio);

    if (this.selectedFile) {
      formData.append('image', this.selectedFile);
    }

    this.profileApi.updateProfile(formData).subscribe({
      next: () => {
        this.refreshProfileAfterUpdate();
      },
      error: err => console.error(err)
    });
  }

  refreshProfileAfterUpdate() {
    this.profileApi.myProfile().subscribe({
      next: (newProfile) => {
        this.user = newProfile;

        this.auth.updateUser(newProfile); // update navbar immediately
        this.previewImage = null;

        const modalEl = document.getElementById('editProfileModal');
        bootstrap.Modal.getInstance(modalEl)?.hide();
      }
    });
  }

  // -------------------------------
  // SELECT PDF FOR NEW POST
  // -------------------------------
  onPdfSelect(event: any) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'application/pdf') return alert('Only PDF allowed');

    this.selectedPdf = file;
    this.selectedPdfName = file.name;
  }

  // -------------------------------
  // CREATE NEW POST
  // -------------------------------
  createPost(e?: Event) {
    e?.preventDefault();

    if (!this.selectedPdf) return alert('Select a PDF first');

    const formData = new FormData();
    formData.append('title', this.newPost.title);
    formData.append('description', this.newPost.description);
    formData.append('file', this.selectedPdf);

    this.profileApi.addPost(formData).subscribe({
      next: (post: any) => {
        this.posts.unshift(post);
        this.totalPosts++;
        this.resetNewPostForm();
      }
    });
  }

  resetNewPostForm() {
    this.newPost = { title: '', description: '' };
    this.selectedPdf = null;
    this.selectedPdfName = null;
  }

  // -------------------------------
  // DELETE POST
  // -------------------------------
  deletePost(id: number) {
    this.profileApi.deleteMyPost(id).subscribe({
      next: () => {
        this.posts = this.posts.filter(p => p.id !== id);
        this.recalculateStats();
        alert('Post deleted');
      }
    });
  }

  recalculateStats() {
    this.totalPosts = this.posts.length;
    this.totalLikes = this.posts.reduce((s, p) => s + p.likes_count, 0);
    this.totalComments = this.posts.reduce((s, p) => s + p.comments.length, 0);
  }

  // -------------------------------
  // EDIT POST
  // -------------------------------
  openEditModal(post: any) {
    this.editPostData = { ...post };
    this.editPdfFile = null;
    this.editPdfName = '';
  }

  onEditPdfSelect(event: any) {
    const file = event.target.files?.[0];
    if (file) {
      this.editPdfFile = file;
      this.editPdfName = file.name;
    }
  }

  updatePost() {
    const formData = new FormData();
    formData.append('title', this.editPostData.title);
    formData.append('description', this.editPostData.description);

    if (this.editPdfFile) {
      formData.append('file', this.editPdfFile);
    }

    this.profileApi.updatePost(this.editPostData.id, formData).subscribe({
      next: (updated: any) => {
        const i = this.posts.findIndex(p => p.id === updated.id);
        if (i !== -1) this.posts[i] = updated;

        bootstrap.Modal.getInstance(document.getElementById('editPostModal'))?.hide();
      }
    });
  }

  // -------------------------------
  // LIKE TOGGLE
  // -------------------------------
  onToggleLike(post: any, id: number) {
    this.profileApi.likePost(id).subscribe({
      next: (res: any) => {
        post.is_liked = res.liked;
        post.likes_count = res.likes_count;

        // update summary
        this.totalLikes = this.posts.reduce((s, p) => s + p.likes_count, 0);
      }
    });
  }

  getRelativeDate(date: string) {
    return formatDistanceToNow(new Date(date), { addSuffix: true });
  }
}
