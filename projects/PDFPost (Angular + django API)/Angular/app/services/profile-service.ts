import { Injectable } from '@angular/core';
import { HttpClient,HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root',
})
export class ProfileService {
  private apiUrl = "http://localhost:8000/api";
  constructor(private http: HttpClient) {}


  toggleVerify(id: number) {
  return this.http.patch(`${this.apiUrl}/users/${id}/toggle-verify/`, {});
}



  myProfile(){
    return this.http.get(`${this.apiUrl}/me/`)
  }


  updateProfile(formData: FormData) {
  return this.http.patch(`${this.apiUrl}/update-profile/`, formData);
}

getProfileById(id:number){
  return this.http.get(`${this.apiUrl}/getprofile/${id}/`);
}

deleteAccount(userId: number) {
  return this.http.delete(`${this.apiUrl}/getprofile/${userId}/`);
}




addPost(formData: FormData) {
  return this.http.post(`${this.apiUrl}/mypost/`, formData);
}

getMyPost(){
  return this.http.get(`${this.apiUrl}/mypost/`);
}
deleteMyPost(id: number){
  return this.http.delete(`${this.apiUrl}/myposts/${id}/`);
}

updatePost(id: number, data: FormData) {
  return this.http.patch(`${this.apiUrl}/myposts/${id}/`, data);
}

// getAllHomePost(){
//   return this.http.get(`${this.apiUrl}/homeposts/`);
// }

getSinglePost(id: number) {
    return this.http.get(`${this.apiUrl}/post/${id}/`);
  }

  //  Add comment to a post
  addComment(postId: number, payload: any) {
    return this.http.post(`${this.apiUrl}/post/${postId}/comment/`, payload);
  }

  deleteComment(postId: number, commentId: number) {
  return this.http.delete(`${this.apiUrl}/post/${postId}/comment/${commentId}/`);
}

  likePost(postId: number){
    return this.http.post(`${this.apiUrl}/posts/${postId}/like/`,{});
  }


   search(query: string): Observable<any> {
    let params = new HttpParams().set("q", query);

    return this.http.get(`${this.apiUrl}/search/`, { params });
  }

  getTopUsersByLikes() {
  return this.http.get(`${this.apiUrl}/top-users-by-likes/`);
}

getTopPosts() {
    return this.http.get<any[]>(`${this.apiUrl}/top-posts/`);
  }

  getHomePosts(page: number, limit: number = 10) {
  return this.http.get<any>(`${this.apiUrl}/home-posts/?page=${page}&limit=${limit}`);
}

}
