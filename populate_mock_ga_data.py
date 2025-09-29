#!/usr/bin/env python3
"""
Script to populate the database with mock GA4 data for testing.
This creates realistic traffic data that can be used for calibration testing.
"""

import os
import random
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import psycopg
from psycopg.rows import dict_row


def generate_mock_ga_events(site_id: str, days: int = 30, sessions_per_day: int = 100) -> List[Dict[str, Any]]:
    """Generate mock GA4 events for testing"""
    
    # Define realistic traffic patterns
    device_categories = ["desktop", "mobile", "tablet"]
    device_weights = [0.6, 0.35, 0.05]
    
    operating_systems = ["Windows", "macOS", "iOS", "Android", "Linux"]
    os_weights = [0.4, 0.2, 0.15, 0.2, 0.05]
    
    source_mediums = [
        "google / organic", "google / cpc", "facebook / social", 
        "twitter / social", "direct / (none)", "email / newsletter",
        "youtube / referral", "reddit / referral"
    ]
    source_weights = [0.35, 0.15, 0.12, 0.08, 0.2, 0.05, 0.03, 0.02]
    
    countries = ["United States", "Canada", "United Kingdom", "Germany", "France", "India", "Australia"]
    country_weights = [0.4, 0.1, 0.15, 0.1, 0.08, 0.12, 0.05]
    
    genders = ["male", "female", "unknown"]
    gender_weights = [0.45, 0.45, 0.1]
    
    age_brackets = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_weights = [0.15, 0.25, 0.2, 0.15, 0.15, 0.1]
    
    # Shopify events with realistic rates
    shopify_events = {
        "page_view": 1.0,  # Every session has page views
        "view_item_list": 0.6,  # 60% of sessions view product lists
        "view_item": 0.4,  # 40% view specific products
        "add_to_cart": 0.08,  # 8% add to cart
        "begin_checkout": 0.012,  # 1.2% begin checkout
        "purchase": 0.003  # 0.3% complete purchase
    }
    
    events = []
    base_time = datetime.now(timezone.utc) - timedelta(days=days)
    
    for day in range(days):
        day_start = base_time + timedelta(days=day)
        
        for session_num in range(sessions_per_day):
            # Generate session characteristics
            session_id = str(uuid.uuid4())
            device_category = random.choices(device_categories, weights=device_weights)[0]
            operating_system = random.choices(operating_systems, weights=os_weights)[0]
            source_medium = random.choices(source_mediums, weights=source_weights)[0]
            country = random.choices(countries, weights=country_weights)[0]
            gender = random.choices(genders, weights=gender_weights)[0]
            age_bracket = random.choices(age_brackets, weights=age_weights)[0]
            
            # Session start time (random within the day)
            session_start = day_start + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            # Generate events for this session
            session_events = []
            
            # Always start with page_view
            session_events.append({
                "event_name": "page_view",
                "event_timestamp": session_start,
                "page_location": f"https://{site_id}/",
                "page_referrer": "" if source_medium == "direct / (none)" else f"https://referrer.com",
                "engagement_time_msec": random.randint(1000, 30000),
                "device_category": device_category,
                "operating_system": operating_system,
                "session_source_medium": source_medium,
                "user_age_bracket": age_bracket,
                "new_vs_returning": "new" if random.random() < 0.7 else "returning",
                "gender": gender,
                "geo_country": country,
                "geo_region": random.choice(["California", "Texas", "New York", "Florida", "Illinois"]) if country == "United States" else "",
                "custom_event": {}
            })
            
            # Generate additional events based on probabilities
            current_time = session_start
            for event_name, probability in shopify_events.items():
                if event_name == "page_view":
                    continue  # Already added
                
                if random.random() < probability:
                    current_time += timedelta(seconds=random.randint(30, 300))
                    engagement_time = random.randint(500, 15000)
                    
                    # Add some realistic backtracking and form errors
                    custom_event = {}
                    if random.random() < 0.1:  # 10% chance of backtrack
                        custom_event["backtrack"] = 1
                    if random.random() < 0.05:  # 5% chance of form error
                        custom_event["form_error"] = 1
                    
                    session_events.append({
                        "event_name": event_name,
                        "event_timestamp": current_time,
                        "page_location": f"https://{site_id}/products" if "item" in event_name else f"https://{site_id}/checkout",
                        "page_referrer": session_events[0]["page_location"],
                        "engagement_time_msec": engagement_time,
                        "device_category": device_category,
                        "operating_system": operating_system,
                        "session_source_medium": source_medium,
                        "user_age_bracket": age_bracket,
                        "new_vs_returning": session_events[0]["new_vs_returning"],
                        "gender": gender,
                        "geo_country": country,
                        "geo_region": session_events[0]["geo_region"],
                        "custom_event": custom_event
                    })
            
            # Add all events from this session
            for event in session_events:
                events.append({
                    "session_id": session_id,
                    **event
                })
    
    return events


def create_ga_events_table(conn):
    """Create the ga_events table if it doesn't exist"""
    with conn.cursor() as cur:
        # Drop existing table if it exists
        cur.execute("DROP TABLE IF EXISTS ga_events CASCADE")
        
        cur.execute("""
            CREATE TABLE ga_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                event_timestamp TIMESTAMPTZ NOT NULL,
                page_location TEXT,
                page_referrer TEXT,
                engagement_time_msec INTEGER DEFAULT 0,
                device_category TEXT,
                operating_system TEXT,
                session_source_medium TEXT,
                user_age_bracket TEXT,
                new_vs_returning TEXT,
                gender TEXT,
                geo_country TEXT,
                geo_region TEXT,
                custom_event JSONB DEFAULT '{}'
            )
        """)
        
        # Create indexes separately
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ga_events_session_id 
            ON ga_events (session_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ga_events_timestamp 
            ON ga_events (event_timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ga_events_event_name 
            ON ga_events (event_name)
        """)
    conn.commit()


def insert_ga_events(conn, events: List[Dict[str, Any]]):
    """Insert GA events into the database"""
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("DELETE FROM ga_events")
        
        # Insert new events in batches
        batch_size = 1000
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            
            # Prepare batch insert
            values = []
            for event in batch:
                values.append((
                    event["session_id"],
                    event["event_name"],
                    event["event_timestamp"],
                    event["page_location"],
                    event["page_referrer"],
                    event["engagement_time_msec"],
                    event["device_category"],
                    event["operating_system"],
                    event["session_source_medium"],
                    event["user_age_bracket"],
                    event["new_vs_returning"],
                    event["gender"],
                    event["geo_country"],
                    event["geo_region"],
                    json.dumps(event["custom_event"])  # Convert dict to JSON string
                ))
            
            cur.executemany("""
                INSERT INTO ga_events (
                    session_id, event_name, event_timestamp, page_location, page_referrer,
                    engagement_time_msec, device_category, operating_system, session_source_medium,
                    user_age_bracket, new_vs_returning, gender, geo_country, geo_region, custom_event
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, values)
        
        conn.commit()


def main():
    """Main function to populate the database with mock GA data"""
    dsn = os.getenv("NEON_DATABASE_URL")
    if not dsn:
        print("Error: NEON_DATABASE_URL environment variable not set")
        return
    
    site_id = "hijabkart.in"  # Default test site
    days = 30
    sessions_per_day = 100
    
    print(f"Generating mock GA data for {site_id}...")
    print(f"Days: {days}, Sessions per day: {sessions_per_day}")
    
    # Generate mock events
    events = generate_mock_ga_events(site_id, days, sessions_per_day)
    print(f"Generated {len(events)} events")
    
    # Connect to database
    print("Connecting to database...")
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        # Create table
        print("Creating ga_events table...")
        create_ga_events_table(conn)
        
        # Insert events
        print("Inserting events into database...")
        insert_ga_events(conn, events)
        
        # Verify insertion
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM ga_events")
            count = cur.fetchone()["count"]
            print(f"Inserted {count} events into database")
            
            # Show some sample data
            cur.execute("""
                SELECT event_name, COUNT(*) as count 
                FROM ga_events 
                GROUP BY event_name 
                ORDER BY count DESC
            """)
            print("\nEvent distribution:")
            for row in cur.fetchall():
                print(f"  {row['event_name']}: {row['count']}")


if __name__ == "__main__":
    main()
